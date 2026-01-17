import os
import sys
import logging
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from src.schemas.scribe_schema import ScribeSchema
from src.schemas.input_schema import InputSchema
from src.state.agent_state import AgentState
from dotenv import load_dotenv
load_dotenv()

# Logger Configuration
logger = logging.getLogger(__name__)

SEED = 42

# Load Model and Prompt (Global Scope = Load Once)
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.1",
    temperature=0,
    seed=SEED,
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

def normalize_vitals(data: dict) -> dict:
    """
    Normalização forçada de dados vitais.
    1. Converte Fahrenheit -> Celsius
    2. Garante consistência Neuro (AVPU vs GCS)
    """
    vitals = data.get("clinical", {}).get("vitals", {})
    
    # --- FIX 1: TERMODINÂMICA (Fahrenheit Killer) ---
    temp = vitals.get("temperature")
    if temp is not None and isinstance(temp, (int, float)):
        # Se for maior que 50, é impossível ser Celsius (seria morte)
        if temp > 50:
            celsius = round((temp - 32) * 5/9, 1)
            print(f"🌡️ AUTO-FIX: Convertendo Temp {temp}F para {celsius}C")
            vitals["temperature"] = celsius
            
    # --- FIX 2: NEUROCONSISTÊNCIA ---
    # Se GCS é 15, AVPU tem de ser Alert
    if vitals.get("gcs") == 15:
        if vitals.get("acvpu") in ["Voice", "Pain", "Unresponsive"]:
            vitals["acvpu"] = "Alert"
            vitals["avpu"] = "Alert"
            
    # Se AVPU é Alert, ACVPU não pode ser Voice
    if vitals.get("avpu") == "Alert" and vitals.get("acvpu") == "Voice":
        vitals["acvpu"] = "Alert"

    return data

def enforce_neuro_consistency(data: dict) -> dict:
    """Corrige alucinações de ACVPU baseadas em GCS e AVPU."""
    vitals = data.get("clinical", {}).get("vitals", {})
    
    # Se GCS é 15 (Máximo), o paciente NUNCA pode ser 'Voice' ou 'Pain'
    if vitals.get("gcs") == 15:
        if vitals.get("acvpu") in ["Voice", "Pain", "Unresponsive"]:
            print(f"🔧 AUTO-FIX: Corrigindo ACVPU de '{vitals['acvpu']}' para 'Alert' baseada em GCS 15.")
            vitals["acvpu"] = "Alert"
            vitals["avpu"] = "Alert"

    # Se AVPU é 'Alert', ACVPU não pode ser 'Voice'
    if vitals.get("avpu") == "Alert" and vitals.get("acvpu") == "Voice":
        print(f"🔧 AUTO-FIX: Corrigindo inconsistência AVPU(Alert) vs ACVPU(Voice).")
        vitals["acvpu"] = "Alert"
        
    return data

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
    
    print("\n--- ✍️ NODE: SCRIBE ---")
    sys.stdout.flush()
    
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
        print(f"⏳ Calling LLM for Extraction (Input Size: {len(input_text)} chars)...")
        sys.stdout.flush()
        
        # LLM Call - returns Pydantic object
        # Reduced timeout to 60s to fail fast if stuck
        response = scribe_model.invoke(messages)

        # Convert to dict if necessary (handles both Pydantic models and raw dicts)
        # if hasattr(response, "dict"):
        #     structured_data = response.dict()
        # elif hasattr(response, "model_dump"):
        #     structured_data = response.model_dump()
        # else:
        #     structured_data = response

        structured_data = response.model_dump()
        structured_data = normalize_vitals(structured_data)

        structured_data = enforce_neuro_consistency(structured_data)
        
        logger.info(f"Extraction success for ID {input_data.subject_id}")
        print(f"{messages}")
        print(f"[ASSISTANT]:")
        print(structured_data)

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
