import re
import logging
import os
from typing import Dict, Any, cast
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from src.schemas.scribe_schema import RawScribeOutput, VitalsSchema, ClinicalSchema
from src.state.agent_state import AgentState
from src.utils.vitals_normalizer import normalize_temperature
from src.utils.run_with_timeout import run_with_timeout
from dotenv import load_dotenv

load_dotenv()
# Configuração de Logs
logger = logging.getLogger(__name__)

class ScribeAgent:
    """
    The Scribe Agent is responsible for extracting structured clinical data from unstructured text.

    Architecture Principle: Single Responsibility Principle (SRP).
    This agent encapsulates the logic for:
    1.  Loading the specific system prompt for clinical extraction.
    2.  Invoking the LLM with structured output constraints (Pydantic).
    3.  Mapping raw LLM outputs to the domain model (ClinicalSchema), including normalization (e.g., AVPU mapping).
    """

    def __init__(self, model: BaseChatModel, prompt_path: str = "prompts/scribe_prompt.md"):
        """
        Initializes the ScribeAgent with a language model and a prompt file.

        Args:
            model (BaseChatModel): The LangChain chat model instance (e.g., ChatOllama) to be used for extraction.
            prompt_path (str): Relative path to the markdown file containing the system prompt. Defaults to "prompts/scribe_prompt.md".
        """
        self.model = model
        self.system_prompt = self._load_prompt(prompt_path)
        preview = self.system_prompt[:120].replace('\n', ' ')
        logger.debug(f"Prompt loaded: {preview} ...")
        self.structured_model = self.model.with_structured_output(RawScribeOutput)

    def _load_prompt(self, path: str) -> str:
        """
        Safely loads the system prompt from a file.

        Args:
            path (str): The file path to the prompt.

        Returns:
            str: The content of the prompt file.

        Raises:
            RuntimeError: If the prompt file cannot be found.
        """
        try:
            full_path = os.path.join(os.getcwd(), path)
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"CRITICAL: Prompt file not found at {path}")
            raise RuntimeError(f"Scribe system prompt missing: {path}")

    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the main extraction pipeline for the Scribe Agent.

        Workflow:
        1.  **Input Retrieval**: Extracts the 'clinical_text' from the AgentState.
        2.  **LLM Invocation**: Sends the text to the LLM using the loaded system prompt and structured output schema (`RawScribeLLM`).
        3.  **Data Normalization**: 
            - Maps ACVPU to AVPU.
            - Normalizes temperature values.
        4.  **Domain Mapping**: Converts the raw LLM output into the strict `ClinicalSchema` domain object.

        Args:
            state (AgentState): The current state of the workflow.

        Returns:
            Dict[str, Any]: A dictionary of state updates, including:
                - `extracted_data`: The `ClinicalSchema` object with vitals and chief complaint.
                - `validation_errors`: A list of any validation errors (empty on success).
                - `attempts`: Incrementing counter for retry logic.
                - `is_success`: Boolean flag indicating operation success.
        """

        # 1. Get the input text
        clinical_text = ""
        input_data = state.get("input")
        attempts = state.get("attempts", 0)
        validation_messages = state.get("validation_messages", [])

        if input_data:
             if hasattr(input_data, "raw_text"):
                 clinical_text = input_data.raw_text
             elif isinstance(input_data, dict):
                 clinical_text = input_data.get("raw_text", "")

        if not clinical_text:
            logger.error("❌Scribe received empty clinical text.")
            return {"error": "Empty input text"}

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Analyze the following clinical text and extract vital signs:\n\n--- BEGIN TEXT ---\n{clinical_text}\n--- END TEXT ---")
        ]

        if validation_messages and attempts > 0:
            logger.info(f"🔄 Scribe retrying with feedback ({len(validation_messages)} errors detected).")
            
            # Adiciona o histórico de erros como mensagens do 'usuário' (simulando um supervisor reclamando)
            messages.extend(validation_messages)
            
            # Adiciona uma instrução final de reforço
            reinforcement = (
                "CRITICAL: The previous extraction contained the errors listed above. "
                "Review the text again carefully. Do NOT repeat the same mistakes. "
                "Ensure numbers are not concatenated (e.g., '12080' is wrong, '120/80' is correct)."
            )
            messages.append(HumanMessage(content=reinforcement))

        try:
            # 2. Extract data with the Scribe Agent
            logger.info("✍--- NODE: SCRIBE ---")
            logger.debug("Extracting data from clinical note...")
            raw_response = run_with_timeout(self.structured_model.invoke, messages, timeout=180, retries=3)
            response: RawScribeOutput = cast(RawScribeOutput, raw_response)
            logger.debug(f"Raw LLM response: {response}")

            if response.span_format == "NOT_FOUND":
                if re.search(r'(?i)\b(VITALS?|BP \d{2,3}/\d{2,3})\b', clinical_text):
                    logger.warning("Vitals exist in text but span selection failed. Forcing retry.")
                    return {
                        "is_success": False,
                        "attempts": attempts + 1,
                        "validation_errors": ["Vitals exist but incorrect span selected."]
                    }
                
            
            
            # 3. Maps ACVPU to AVPU
            logger.debug("Mapping ACVPU to AVPU...")
            
            # 4. Convert data to ClinicalSchema
            logger.debug("Converting to ClinicalSchema...")
            clinical_output = response.to_domain()
            domain_vitals = clinical_output.vitals

            logger.info(f"Extraction Successful. HR: {domain_vitals.heartrate}, BP: {domain_vitals.sbp}/{domain_vitals.dbp}, Temp: {domain_vitals.temperature}, O2Sat: {domain_vitals.o2sat}")

            return {
                "extracted_data": clinical_output,
                "validation_errors": [],
                "attempts": state.get("attempts", 0) + 1,
                "is_success": True
            }

        except ValidationError as e:
            logger.error(f"Validation Error in Scribe: {str(e)}")
            return {
                "validation_errors": [str(e)],
                "attempts": state.get("attempts", 0) + 1,
                "is_success": False
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in Scribe Agent: {e}", exc_info=True)
            return {
                "error": str(e),
                "is_success": False
            }

# if __name__ == "__main__":
#     llm = ChatOllama(
#         model="llama3.1:8b",
#         temperature=0,
#         seed=42,
#     )
#     agent = ScribeAgent(llm)
#     test_messages = [
#         SystemMessage(content="Say 'pong'"),
#         HumanMessage(content="ping")
#     ]
#     print(run_with_timeout(llm.invoke, test_messages, timeout=15))