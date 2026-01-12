from langchain_core.messages import HumanMessage
from src.state.agent_state import AgentState
from pydantic import ValidationError
from src.schemas.scribe_schema import ScribeSchema

def validator_node(state: AgentState) -> AgentState:
    print("--- 🛡️ NODE: VALIDATOR ---")

    data = state.get("extracted_data")

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
        acuity = vitals.pain
        gcs = vitals.gcs

        if temp and (temp <= 25.0 or temp >= 45.0):
            state["validation_errors"] = [f"Temperature value {temp} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The body temperature {temp} seems incorrect.
                Check the original text and correct it.""")]
            
        if hr and (hr < 0 or hr > 300):
            state["validation_errors"] = [f"Heart rate value {hr} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The heart rate {hr} seems incorrect.
                Check the original text and correct it.""")]
           
        if rr and (rr <= 0 or rr > 60):
            state["validation_errors"] = [f"Respiratory rate value {rr} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The Respiratory rate {rr} seems incorrect.
                Check the original text and correct it.""")]
            
        if o2sat and (0 > o2sat or o2sat > 100):
            state["validation_errors"] = [f"Oxygen (0_2) saturation value {o2sat} is impossible."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The oxygen saturation {o2sat} is incorrect.
                Check the original text and correct it.""")]

        if sbp and (sbp > 250 or sbp < 40):
            state["validation_errors"] = [f"SBP value {sbp} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The systolic blood pressure {sbp} seems incorrect.
                Check the original text and correct it.""")]
            
        if dbp and (dbp < 10 or dbp > 300):
            state["validation_errors"] = [f"DBP value {dbp} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The diastolic blood pressure {sbp} seems incorrect.
                Check the original text and correct it.""")]
            
        if pain and (pain < 0 or pain > 10):
            state["validation_errors"] = [f"Pain value {pain} is technical impossible."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The pain value {pain} is incorrect.
                Check the original text and correct it.""")]
            
        if pain and (pain < 0 or pain > 10):
            state["validation_errors"] = [f"Pain value {pain} is technical impossible."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The pain value {pain} is incorrect.
                Check the original text and correct it.""")]
            
        if acuity and (acuity < 1 or acuity > 5):
            state["validation_errors"] = [f"Acuity value {acuity} is impossible."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The pain value {acuity} is incorrect.
                Check the original text and correct it.""")]
            
        if gcs and (gcs < 3 or gcs > 15):
            state["validation_errors"] = [f"GCS value {gcs} is impossible."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The Glasgow Coma Scale value {gcs} is incorrect.
                Check the original text and correct it.""")]
    
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

        if potassium and (potassium < 0.5 or potassium > 12.0):
            state["validation_errors"] = [f"Potassium concentration {potassium} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The potassium concentration {potassium} seems incorrect.
                Check the original text and correct it.""")]
            
        if sodium and (sodium < 80.0 or sodium > 200.0):
            state["validation_errors"] = [f"Sodium concentration {sodium} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The sodium concentration {sodium} seems incorrect.
                Check the original text and correct it.""")]
            
        if creatinine and (creatinine < 0.1 or creatinine > 40.0):
            state["validation_errors"] = [f"Creatinine concentration {creatinine} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The creatinine concentration {creatinine} seems incorrect.
                Check the original text and correct it.""")]
            
        if wbc and (wbc < 0 or wbc > 500):
            state["validation_errors"] = [f"WBC concentration {wbc} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The White Blood Cell concentration {wbc} seems incorrect.
                Check the original text and correct it.""")]
        
        if platelets and (platelets < 0 or platelets > 2000):
            state["validation_errors"] = [f"Platelets concentration {platelets} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The platelets concentration {platelets} seems incorrect.
                Check the original text and correct it.""")]
        
        if inr and (inr < 0.5 or inr > 30.0):
            state["validation_errors"] = [f"INR value {inr} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The International Normalized Ratio {inr} seems incorrect.
                Check the original text and correct it.""")]
            
        if albumin and (albumin < 0.5 or albumin > 7.0):
            state["validation_errors"] = [f"Albumin concentration {albumin} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The albumin concentration {albumin} seems incorrect.
                Check the original text and correct it.""")]
            
        if bilirubin and (bilirubin < 0.1 or bilirubin > 80):
            state["validation_errors"] = [f"Bilirubin concentration {bilirubin} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The bilirubin concentration {bilirubin} seems incorrect.
                Check the original text and correct it.""")]
            
        if ast and (ast < 0 or ast > 20000):
            state["validation_errors"] = [f"AST concentration {ast} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The Aspartate Aminotransferase concentration {ast} seems incorrect.
                Check the original text and correct it.""")]

        if alt and (alt < 0 or alt > 20000):
            state["validation_errors"] = [f"ALT concentration {alt} is physiologically improbable."]
            state["validation_messages"] = [HumanMessage(
                content=f"""[CRITICAL ERROR]: The Alanine Aminotransferase concentration {alt} seems incorrect.
                Check the original text and correct it.""")]
            
    state["validation_errors"] = []
    return state

def validator_router(state: AgentState) -> str:
    if state.get("validation_errors"):
        return "scribe"
    return "supervisor"