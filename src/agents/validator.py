"""Validator agent for physiological plausibility checking.

This module applies rule-based validation on extracted vital signs to
detect physiologically impossible or implausible values, triggering
retry logic in the Scribe agent when errors are found.
"""

import logging
from typing import Any, Dict, List, Literal

from langchain_core.messages import HumanMessage

from src.state.agent_state import AgentState
from src.schemas.scribe_schema import ClinicalSchema

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
    logger.info("--- NODE: VALIDATOR ---")

    data = state.get("extracted_data")

    if not data or not hasattr(data, "vitals"):
        return {"validation_errors": ["No extracted data found."]}

    if isinstance(data, dict):
        try:
            data = ClinicalSchema(**data)
        except Exception as e:
            logger.error("Schema validation error: %s", e)
            return {"validation_errors": [f"Schema Validation Error: {e}"]}

    # 1. Verify if there are any errors.
    errors: List[str] = []
    messages: List[HumanMessage] = []

    if not data:
        logger.error("No data extracted.")
        return {"validation_errors": ["No extracted data found"]}

    vitals = data.vitals

    if not vitals:
        logger.error("No vitals found.")
        return {"validation_errors": ["No vitals found"]}


    if vitals:
        sbp = vitals.sbp
        dbp = vitals.dbp
        temp = vitals.temperature
        hr = vitals.heartrate
        rr = vitals.resprate
        o2sat = vitals.o2sat
 
        logger.debug("Validating vitals...")
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
                      msg = f"""
                      Temperature value {temp} is physiologically improbable
                      (Celsius range 25-45)."""
                      errors.append(msg)
                      messages.append(HumanMessage(
                          content=f"[CRITICAL ERROR]: {msg} Check units."
                          ))
            # Check for too low values
            elif temp <= 25.0:
                msg = f"""Temperature value {temp} is physiologically improbable
                (Celsius range 25-45)."""
                errors.append(msg)
                messages.append(HumanMessage(
                    content=f"[CRITICAL ERROR]: {msg} Check units."
                ))

        if hr is not None and (hr < 0 or hr > 300):
            msg = f"Heart rate value {hr} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(
                content=f"[CRITICAL ERROR]: {msg} Check original text."
            ))
           
        if rr is not None and (rr <= 0 or rr > 60):
            msg = f"Respiratory rate value {rr} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(
                content=f"[CRITICAL ERROR]: {msg} Check original text."
            ))
            
        if o2sat is not None and (0 > o2sat or o2sat > 100):
            msg = f"Oxygen (O2) saturation value {o2sat} is impossible (0-100)."
            errors.append(msg)
            messages.append(HumanMessage(
                content=f"[CRITICAL ERROR]: {msg} Check original text."
            ))

        if sbp is not None and (sbp > 300 or sbp < 40): 
            msg = f"SBP value {sbp} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(
                content=f"[CRITICAL ERROR]: {msg} Check original text."
            ))
            
        if dbp is not None and (dbp < 10 or dbp > 200): 
            msg = f"DBP value {dbp} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(
                content=f"[CRITICAL ERROR]: {msg} Check original text."
            ))
        
        all_none = all(v is None for v in [vitals.heartrate, vitals.resprate, vitals.temperature, vitals.o2sat, vitals.sbp])
    
        if all_none:
            msg = """
            Extraction failed to retrieve ANY numeric vitals.
            The selected span was likely text-only or invalid."""
            errors.append(msg)
            messages.append(HumanMessage(
                content=f"""[CRITICAL ERROR]: {msg} Please re-read the text
                and find the section with ACTUAL NUMBERS (e.g., 'BP 120/80', 'HR 80')."""
            ))
            logger.warning(msg)
        
        updates = {}
        output_update = {
            "validation_errors": errors,
            "validation_messages": messages,
        }
        
        if errors and state.get("attempts", 0) >= 2:
            if vitals.sbp is not None and (vitals.sbp > 300 or vitals.sbp < 40):
                logger.warning("Scrubbing invalid SBP: %s", vitals.sbp)
                updates['sbp'] = None
                msg = f"""SBP value {vitals.sbp} is physiologically improbable
                and was removed."""
                errors.append(msg)
                messages.append(HumanMessage(
                    content=f"[CRITICAL ERROR]: {msg} Check original text."))

            if vitals.dbp is not None and (vitals.dbp > 200 or vitals.dbp < 10):
                logger.warning("Scrubbing invalid DBP: %s", vitals.dbp)
                updates['dbp'] = None
                msg = f"""DBP value {vitals.dbp} is physiologically improbable
                and was removed."""
                errors.append(msg)
                messages.append(HumanMessage(
                    content=f"[CRITICAL ERROR]: {msg} Check original text."
                ))

            if vitals.heartrate is not None and (vitals.heartrate > 220 or vitals.heartrate < 20):
                logger.warning("Scrubbing invalid HR: %s", vitals.heartrate)
                updates['heartrate'] = None
                
            if updates:
                new_vitals = vitals.model_copy(update=updates)
                logger.info(
                    "Vitals after scrub: SBP=%s, DBP=%s, HR=%s",
                    new_vitals.sbp, new_vitals.dbp, new_vitals.heartrate,
                )
                
                new_clinical_data = data.model_copy(update={"vitals": new_vitals})
                data = new_clinical_data
                output_update["extracted_data"] = new_clinical_data
                state["extracted_data"] = new_clinical_data
                state.update({"extracted_data": new_clinical_data})

            if errors is None and messages is None:
                logger.info("Data validated successfully!")

    return {
        "extracted_data": data.model_dump(),
        "validation_errors": errors,
        "validation_messages": messages
    }