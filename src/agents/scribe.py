import logging
import os
from typing import Dict, Any, cast
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from src.schemas.scribe_schema import RawScribeLLM, VitalsSchema, ClinicalSchema
from src.state.agent_state import AgentState
from src.utils.vitals_normalizer import normalize_temperature

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
            model (BaseChatModel): The LangChain chat model instance (e.g., ChatOllama, ChatOpenAI) to be used for extraction.
            prompt_path (str): Relative path to the markdown file containing the system prompt. Defaults to "prompts/scribe_prompt.md".
        """
        self.model = model
        self.system_prompt = self._load_prompt(prompt_path)
        self.structured_model = self.model.with_structured_output(RawScribeLLM)

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

    def _map_avpu(self, raw_acvpu: str) -> str:
        """
        Maps the raw ACVPU (Alert, Confusion, Voice, Pain, Unresponsive) scale to the standard AVPU scale.
        
        Business Logic:
        - 'Confusion' is mapped to 'Alert' (as per some triage protocols where confusion is a status of an alert patient, though this can be customized).
        - 'Alert', 'Voice', 'Pain', 'Unresponsive' map to themselves.

        Args:
            raw_acvpu (str): The raw mental status string extracted by the LLM.

        Returns:
            str: The normalized AVPU status. Defaults to 'Alert' if input is empty or unknown.
        """
        if not raw_acvpu:
            return "Alert"
            
        mapping = {
            "Confusion": "Alert",
            "Alert": "Alert",
            "Voice": "Voice",
            "Pain": "Pain",
            "Unresponsive": "Unresponsive"
        }
        return mapping.get(raw_acvpu, "Alert") # Default fallback

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
            HumanMessage(content=f"INPUT NOTES:\n{clinical_text}")
        ]

        try:
            # 2. Extract data with the Scribe Agent
            logger.info("✍--- NODE: SCRIBE ---")
            logger.debug("Extracting data from clinical note...")
            response: RawScribeLLM = cast(RawScribeLLM, self.structured_model.invoke(messages))
            vitals = response.vitals
            
            # 3. Maps ACVPU to AVPU
            logger.debug("Mapping ACVPU to AVPU...")
            final_acvpu = vitals.acvpu
            avpu_mapped = self._map_avpu(final_acvpu)
            
            # 4. Convert data to ClinicalSchema
            logger.debug("Converting to ClinicalSchema...")
            domain_vitals = VitalsSchema(
                heartrate=vitals.heartrate,
                resprate=vitals.resprate,
                temperature=normalize_temperature(vitals.temperature) if vitals.temperature else None, # O validator do Pydantic já deve ter normalizado
                o2sat=vitals.o2sat,
                sbp=vitals.sbp,
                dbp=vitals.dbp,
                avpu=avpu_mapped,
                acvpu=final_acvpu,
                supplemental_oxygen=vitals.supplemental_oxygen or False,
                acuity=None 
            )

            clinical_output = ClinicalSchema(
                chief_complaint=response.chief_complaint or "Not reported",
                vitals=domain_vitals
            )
            # print(clinical_output)

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
