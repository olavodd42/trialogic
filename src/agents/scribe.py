import logging
from typing import Dict, Any, TypedDict, Optional, cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.schemas.scribe_schema import ScribeSchema
from src.schemas.input_schema import InputSchema
from src.state.agent_state import AgentState
from dotenv import load_dotenv
load_dotenv()

# Configuração de Logs (Essencial para TCC e Debug)
logger = logging.getLogger(__name__)


# Carregamento do Modelo e Prompt (Escopo Global = Carrega 1 vez)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
scribe_model = llm.with_structured_output(ScribeSchema)

try:
    with open("prompts/system_prompt.md", "r", encoding="utf-8") as f:
        SCRIBE_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    # Fail-fast: Se não tem prompt, o sistema nem deve subir
    raise RuntimeError("Critical: system_prompt.md not found.")

def scribe_node(state: AgentState) -> Dict:
    """
    The 'Scribe' Node: First step in the processing pipeline. 
    
    This agent uses a Large Language Model (LLM) to parse raw, unstructured clinical text into a 
    structured, schema-validated JSON format (ScribeSchema). 
    It incorporates feedback loops: if a previous attempt failed validation, the error message 
    is injected back into the prompt to guide the model's correction.

    Args:
        state (AgentState): The current state of the agent workflow, containing the raw input and any error metadata.

    Returns:
        dict: The updated state with the extracted structured data (or error details).
    """
    print("--- ✍️ NODE: SCRIBE ---")
    
    input_data = state.get("input")
    input_text = input_data.raw_text
    errors = state.get("errors", [])

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
        # A chamada da LLM
        structured_data = cast(ScribeSchema, llm.invoke(messages))
        
        logger.info(f"Extraction success for ID {input_data.subject_id}")

        return {
            "input": input_data,
            "extracted_data": structured_data.model_dump(),
            "validation_error": None,
            "attempts": attempts + 1,
            "risk_score_report": None  # Inicialmente None, será preenchido depois
        }
    
    except Exception as e:
        logger.error(f"LLM Validation failed: {e}")
        # Graceful degradation: Não quebre o grafo, marque como erro
        return {
            "input": input_data,
            "extracted_data": None,
            "validation_error": str(e),
            "attempts": attempts + 1,
            "risk_score_report": None
        }
