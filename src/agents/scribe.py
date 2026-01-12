import logging
from typing import Dict, Any, TypedDict, Optional, cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.schemas.scribe_schema import ScribeOutputSchema
from src.schemas.input_schema import InputSchema
from src.state.agent_state import AgentState
from dotenv import load_dotenv
load_dotenv()

# Configuração de Logs (Essencial para TCC e Debug)
logger = logging.getLogger(__name__)


# Carregamento do Modelo e Prompt (Escopo Global = Carrega 1 vez)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
scribe_model = llm.with_structured_output(ScribeOutputSchema)

try:
    with open("prompts/system_prompt.md", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_CONTENT = f.read()
except FileNotFoundError:
    # Fail-fast: Se não tem prompt, o sistema nem deve subir
    raise RuntimeError("Critical: system_prompt.md not found.")

def scribe_node(state: AgentState) -> dict:
    """
    The 'Scribe' Node: First step in the processing pipeline. 
    
    This agent uses a Large Language Model (LLM) to parse raw, unstructured clinical text into a 
    structured, schema-validated JSON format (ScribeOutputSchema). 
    It incorporates feedback loops: if a previous attempt failed validation, the error message 
    is injected back into the prompt to guide the model's correction.

    Args:
        state (AgentState): The current state of the agent workflow, containing the raw input and any error metadata.

    Returns:
        dict: The updated state with the extracted structured data (or error details).
    """
    print("--- [SCRIBE]: Starting structured extraction ---")
    
    input_data: InputSchema = state["input"]
    error = state.get("validation_error", None)
    attempts = state.get("attempts", 0)

    system_prompt = SYSTEM_PROMPT_CONTENT
    user_prompt = input_data.raw_text

    if error:
        user_prompt += f"\n\nNote that the previous attempt resulted in the following validation error: \
            {error}\nPlease correct the output accordingly."

    try:
        # A chamada da LLM
        structured_data = cast(ScribeOutputSchema, scribe_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]))
        
        logger.info(f"Extraction success for ID {input_data.subject_id}")
        
        # RETORNO PARA LANGGRAPH:
        # Retornamos apenas o que queremos atualizar no estado global
        # Convertemos para dict para serialização fácil no estado
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
    
def validator(state: AgentState) -> str:
    """
    Conditional logic to determine the next step in the workflow graph.
    
    Checks if the Scribe's extraction resulted in a validation error.
    
    Returns:
        str: "next_node" if extraction was successful (proceed to Risk Assessment), 
             or "continue" to retry the Scribe step with error feedback.
    """
    error = state.get("validation_error", None)
    attempts = state.get("attempts", 0)
    MAX_RETRIES = 3
    
    if not error:
        return "next_node"
    return "continue"
