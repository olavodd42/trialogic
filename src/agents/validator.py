from typing import List, Literal, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from src.state.agent_state import AgentState
from src.schemas.scribe_schema import RawScribeLLM
import logging

logger = logging.getLogger(__name__)

def validator_router(state: AgentState) -> Literal["scribe", "supervisor"]:
    """
    Router determining the path after validation.
    
    Returns:
        'scribe' if there are validation errors (and attempts allow retry).
        'supervisor' if validation passed (to let supervisor decide next step).
    """
    errors = state.get("validation_errors", [])
    attempts = state.get("attempts", 0)

    
    if errors and attempts < 3:
        return "scribe"
    return "supervisor"

def validator_node(state: AgentState) -> Dict[str, Any]:
    """
    The Validator Node checks the physiological plausibility of the extracted data.

    It performs rule-based validation on vital signs and lab results.
    If values are out of physically possible or likely biological ranges, it flags them as errors.
    This feedback can be used by the Scribe agent to retry extraction or by human reviewers.

    Args:
        state (AgentState): The current state containing extracted data.

    Returns:
        dict: Updates to the state, specifically 'validation_errors' and 'validation_messages'.
    """
    print("--- 🛡️ NODE: VALIDATOR ---")

    data = state.get("extracted_data")
    
    # Ensure data is a Pydantic object for dot notation access
    if isinstance(data, dict):
        try:
            data = RawScribeLLM(**data)
        except Exception as e:
            return {"validation_errors": [f"Schema Validation Error: {e}"]}

    # Initialize error containers
    errors: List[str] = []
    messages: List[HumanMessage] = []

    if not data:
        logger.error("No data extracted")
        return {"validation_errors": ["No extracted data found"]}

    vitals = data.vitals

    if not vitals:
        logger.error("No vitals found")
        return {"validation_errors": ["No vitals found"]}


    if vitals:
        sbp = vitals.sbp
        dbp = vitals.dbp
        temp = vitals.temperature
        hr = vitals.heartrate
        rr = vitals.resprate
        o2sat = vitals.o2sat
        acuity = vitals.acuity 

        # Temperature Normalization and Validation
        if temp is not None:
            # Check for Fahrenheit-like values (e.g. > 45)
            if temp > 45.0:
                 temp_c = (temp - 32) * 5.0 / 9.0
                 # If the converted value is valid, accept it and update
                 if 25.0 <= temp_c <= 45.0:
                      vitals.temperature = round(temp_c, 1)
                      # Log info if needed, but no error
                 else:
                      msg = f"Temperature value {temp} is physiologically improbable (Celsius range 25-45)."
                      errors.append(msg)
                      messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check units."))
            # Check for too low values
            elif temp <= 25.0:
                msg = f"Temperature value {temp} is physiologically improbable (Celsius range 25-45)."
                errors.append(msg)
                messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check units."))

        if hr is not None and (hr < 0 or hr > 300):
            msg = f"Heart rate value {hr} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
           
        if rr is not None and (rr <= 0 or rr > 60):
            msg = f"Respiratory rate value {rr} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
            
        if o2sat is not None and (0 > o2sat or o2sat > 100):
            msg = f"Oxygen (O2) saturation value {o2sat} is impossible (0-100)."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))

        if sbp is not None and (sbp > 300 or sbp < 40): 
            msg = f"SBP value {sbp} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
            
        if dbp is not None and (dbp < 10 or dbp > 200): 
            msg = f"DBP value {dbp} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))

        if acuity is not None and (acuity < 1 or acuity > 5):
            msg = f"Acuity value {acuity} is impossible (ESI is 1-5)."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
            
    return {
        "extracted_data": data.model_dump(),
        "validation_errors": errors,
        "validation_messages": messages
    }