import logging
from typing import Dict, Any, TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.schemas.scribe_output_schema import ScribeOutputSchema
from src.schemas.input_schema import InputSchema
from dotenv import load_dotenv
load_dotenv()

# Configuração de Logs (Essencial para TCC e Debug)
logger = logging.getLogger(__name__)
class AgentState(TypedDict):
    input: InputSchema              # O input bruto da triagem
    extracted_data: Optional[ScribeOutputSchema] # O JSON validado (SinaisVitais)
    validation_error: Optional[str]   # Mensagem de erro do Pydantic (se houver)
    attempts: int

# Carregamento do Modelo e Prompt (Escopo Global = Carrega 1 vez)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
scribe_model = llm.with_structured_output(ScribeOutputSchema)

try:
    with open("prompts/system_prompt.md", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_CONTENT = f.read()
except FileNotFoundError:
    # Fail-fast: Se não tem prompt, o sistema nem deve subir
    raise RuntimeError("Critical: system_prompt.md not found.")

def scribe_node(state: AgentState) -> AgentState:
    """
    Nó 1: The Scribe
    Integration with LLM to extract structured data from unstructured clinical text.
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
        structured_data = scribe_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        logger.info(f"Extraction success for ID {input_data.subject_id}")
        
        # RETORNO PARA LANGGRAPH:
        # Retornamos apenas o que queremos atualizar no estado global
        # Convertemos para dict para serialização fácil no estado
        return {
            "input": input_data,
            "extracted_data": structured_data.model_dump(),
            "validation_error": None,
            "attempts": attempts + 1
        }
    
    except Exception as e:
        logger.error(f"LLM Validation failed: {e}")
        # Graceful degradation: Não quebre o grafo, marque como erro
        return {
            "input": input_data,
            "extracted_data": None,
            "validation_error": str(e),
            "attempts": attempts + 1
        }
    
def validator(state: AgentState) -> str:
    """Function to validate the extracted data."""
    error = state.get("validation_error", None)
    attempts = state.get("attempts", 0)
    MAX_RETRIES = 3
    
    if not error:
        return "next_node"
    return "continue"
