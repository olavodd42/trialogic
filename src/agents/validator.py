from typing import List, Literal, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from src.state.agent_state import AgentState
from src.schemas.scribe_schema import ScribeSchema

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
            data = ScribeSchema(**data)
        except Exception as e:
            return {"validation_errors": [f"Schema Validation Error: {e}"]}

    # Initialize error containers
    errors: List[str] = []
    messages: List[HumanMessage] = []

    vitals = None
    labs = None
    if data and data.clinical:
        vitals = data.clinical.vitals
        labs = data.clinical.labs

    if vitals:
        sbp = vitals.sbp
        dbp = vitals.dbp
        temp = vitals.temperature
        hr = vitals.heartrate
        rr = vitals.resprate
        o2sat = vitals.o2sat
        pain = vitals.pain
        acuity = vitals.acuity 
        gcs = vitals.gcs

        if temp is not None:
            if (temp <= 25.0 or temp >= 45.0):
                converted_temp = round((temp - 32) * 5/9, 1)
                msg = f"Temperature {vitals.temperature} likely Fahrenheit. Auto-converted to {converted_temp} C."
                print(f"⚠️ {msg}")
                vitals.temperature = converted_temp
                state.get("validation_messages", []).append(msg)

            elif temp < 25 or temp > 45:
                msg = f"Temperature value {vitals.temperature} is physiologically improbable (Celsius range 25-45)."
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
            
        if pain is not None and (pain < 0 or pain > 10):
            msg = f"Pain value {pain} is technically impossible (0-10)."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
            
        if acuity is not None and (acuity < 1 or acuity > 5):
            msg = f"Acuity value {acuity} is impossible (ESI is 1-5)."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
            
        if gcs is not None and (gcs < 3 or gcs > 15):
            msg = f"GCS value {gcs} is impossible (3-15)."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
    
    if labs:
        potassium = labs.potassium
        sodium = labs.sodium
        creatinine = labs.creatinine
        wbc = labs.wbc
        platelets = labs.platelets
        inr = labs.inr
        albumin = labs.albumin
        bilirubin = labs.bilirubin
        ast = labs.ast
        alt = labs.alt

        if potassium is not None and (potassium < 0.5 or potassium > 12.0):
            msg = f"Potassium concentration {potassium} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
        
        if sodium is not None and (sodium < 80.0 or sodium > 200.0):
            msg = f"Sodium concentration {sodium} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
        
        if creatinine is not None and (creatinine < 0.1 or creatinine > 40.0):
            msg = f"Creatinine concentration {creatinine} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
            
        if wbc is not None and (wbc < 0 or wbc > 500):
            msg = f"WBC concentration {wbc} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
        
        if platelets is not None and (platelets < 0 or platelets > 2000):
            msg = f"Platelets concentration {platelets} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))

        if inr is not None and (inr < 0.5 or inr > 30.0):
            msg = f"INR value {inr} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
            
        if albumin is not None and (albumin < 0.5 or albumin > 7.0):
            msg = f"Albumin concentration {albumin} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))
            
        if bilirubin is not None and (bilirubin < 0.1 or bilirubin > 80):
            msg = f"Bilirubin concentration {bilirubin} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))

        if ast is not None and (ast < 0 or ast > 20000):
            msg = f"AST concentration {ast} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))

        if alt is not None and (alt < 0 or alt > 20000):
            msg = f"ALT concentration {alt} is physiologically improbable."
            errors.append(msg)
            messages.append(HumanMessage(content=f"[CRITICAL ERROR]: {msg} Check original text."))

    return {
        "validation_errors": errors,
        "validation_messages": messages
    }