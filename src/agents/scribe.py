import os
import logging
import json
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.state.agent_state import AgentState
from src.schemas.scribe_schema import ScribeSchema
from src.schemas.input_schema import InputSchema

load_dotenv()
logger = logging.getLogger(__name__)

# LLM Setup
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.1", 
    temperature=0,
    seed=42
)
scribe_model = llm.with_structured_output(ScribeSchema)

# Load Prompt
prompt_path = os.path.join(os.getcwd(), "prompts", "scribe_prompt.md")
try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        SCRIBE_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    logger.warning("Scribe prompt not found. Using default.")
    SCRIBE_SYSTEM_PROMPT = "You are a clinical extraction specialist."

def scribe_node(state: AgentState) -> Dict[str, Any]:
    """
    Extracts structured clinical entities from the note.
    UPDATED: Hoists 'vitals' to top-level state for easier access by downstream agents.
    """
    print("\n--- ✍️ NODE: SCRIBE ---")
    
    # 1. Get Input and extract text safely
    input_data = state.get("input")
    raw_note = ""
    
    if hasattr(input_data, "raw_text"):
        raw_note = input_data.raw_text
    elif isinstance(input_data, dict) and "raw_text" in input_data:
        raw_note = input_data.get("raw_text", "")
    elif isinstance(input_data, str):
        raw_note = input_data
        
    if not raw_note:
        logger.warning("Scribe received empty input.")
        print("⚠️ Warning: Empty input for Scribe.")
        return {"validation_errors": ["Empty input text"]}
    print(input_data)
    # 2. Invoke LLM
    try:
        messages = [
            SystemMessage(content=SCRIBE_SYSTEM_PROMPT),
            HumanMessage(content=f"Clinical Note: {raw_note}")
        ]
        
        logger.info(f"Calling LLM for Extraction (Input Size: {len(raw_note)} chars)...")
        print(f"⏳ Calling LLM for Extraction (Input Size: {len(raw_note)} chars)...")
        
        response = scribe_model.invoke(messages)

        print(response)
        
        # 3. Process Output
        # Convert Pydantic model to dict
        output_data = response
        # vitals_found = output_data.clinical.vitals
        
        updates = {
            "extracted_data": output_data,
            "validation_errors": [],
            "attempts": state.get("attempts", 0) + 1,
        }
        
        # Log success details
        vitals = output_data.clinical.vitals
        if vitals:
             print(f"✅ Extraction Complete. Vitals Vlaues: {vitals.model_dump()}")
        else:
             print("⚠️ No vitals object found in extracted data.")

        
        return updates

    except Exception as e:
        logger.error(f"Scribe extraction failed: {e}")
        print(f"❌ Scribe Error: {e}")

        return {
            "validation_errors": [str(e)],
            "attempts": state.get("attempts", 0) + 1
        }