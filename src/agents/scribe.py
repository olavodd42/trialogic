import os
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from src.schemas.scribe_schema import ScribeSchema
from src.schemas.input_schema import InputSchema
from src.state.agent_state import AgentState
from dotenv import load_dotenv
load_dotenv()

# Logger Configuration
logger = logging.getLogger(__name__)


# Load Model and Prompt (Global Scope = Load Once)
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key=SecretStr("lm-studio"),
    model="gpt-4o-mini",
    temperature=0
)

scribe_model = llm.with_structured_output(ScribeSchema)

# Load prompt with cross-platform path handling
prompt_path = os.path.join(os.getcwd(), "prompts", "scribe_prompt.md")
try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        SCRIBE_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    # Fail-fast: System cannot start without the prompt
    raise RuntimeError(f"Critical: scribe_prompt.md not found at {prompt_path}.")

def scribe_node(state: AgentState) -> Dict[str, Any]:
    """
    The 'Scribe' Node: First step in the processing pipeline. 
    
    This agent uses a Large Language Model (LLM) to parse raw, unstructured clinical text into a 
    structured, schema-validated JSON format (ScribeSchema). 
    It incorporates feedback loops: if a previous attempt failed validation, the error message 
    is injected back into the prompt to guide the model's correction.

    Args:
        state (AgentState): The current state of the agent workflow.

    Returns:
        dict: The updated state with the extracted structured data (or error details).
    """
    print("--- ✍️ NODE: SCRIBE ---")
    
    input_data = state.get("input")
    # Handle the case where input might not be what we expect, though strict typing suggests it is InputSchema
    if not input_data:
        return {"validation_errors": ["No input data provided."]}

    if hasattr(input_data, "raw_text"):
        input_text = input_data.raw_text
    else:
        raise AttributeError(f"{type(input_data)} has no attribute 'raw_text'.")
    errors = state.get("validation_errors", [])

    messages = [
        SystemMessage(content=SCRIBE_SYSTEM_PROMPT),
        HumanMessage(content=f"Clinical Note: {input_text}")
    ]
    attempts = state.get("attempts", 0)
    if errors:
        print(f"🔄 Scribe retrying due to errors: {errors}")
        error_msg = f"Your previous extraction had critical errors: {errors}. Please fix them and extract again."
        messages.append(HumanMessage(content=error_msg))

    try:
        # LLM Call - returns Pydantic object
        structured_data = scribe_model.invoke(messages)
        
        logger.info(f"Extraction success for ID {input_data.subject_id}")

        return {
            "extracted_data": structured_data,
            "validation_errors": [],
            "attempts": attempts + 1,
        }
    
    except Exception as e:
        logger.error(f"LLM Validation failed: {e}")
        # Graceful degradation: Do not break the graph, mark as error
        return {
            "extracted_data": None,
            "validation_errors": [str(e)],
            "attempts": attempts + 1,
        }
