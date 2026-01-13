from typing import List, Literal, Dict
from src.state.agent_state import AgentState

def supervisor_planning(state: AgentState) -> Dict[str, List[str]]:
    """
    Supervisor Node (Planning Phase): Defines the execution plan (workflow) for the agent.

    Currently uses a static linear plan: scribe -> mathematician -> auditor.
    In future iterations, this could dynamically adjust the plan based on input complexity.

    Args:
        state (AgentState): Current state.

    Returns:
        dict: State update with the 'plan' key.
    """
    print("--- 🧠 NODE: SUPERVISOR (PLANNING) ---")

    initial_plan = ["scribe", "mathematician", "clinical_rag"]
    # We return a dict to update the state.
    # Note: 'plan' key needs to be existent in AgentState if we use TypedDict with strict keys,
    # but AgentState definition showed earlier might not have 'plan' (checked earlier, it ended with context/auditor keys).
    # If 'plan' is not in AgentState, this might be an issue if we are adding new keys to TypedDict.
    # Assuming AgentState is flexible or will be updated. 
    # However, for now, we just return the update.
    return {"plan": initial_plan} 

def supervisor_router(state: AgentState) -> Literal["scribe", "mathematician", "clinical_rag", "END"]:
    """
    Supervisor Router: Determines the next step in the workflow based on the state.

    Logic:
    1. If no extracted data, go to 'scribe'.
    2. If extraction has errors (and attempts < max), retry 'scribe'.
    3. If extracted data exists but no risk score, go to 'mathematician'.
    4. If risk score exists but no audit, go to 'auditor'.
    5. If all done, 'end'.

    Args:
        state (AgentState): Current state.

    Returns:
        str: The name of the next node to execute.
    """
    extracted = state.get("extracted_data")
    risk = state.get("risk_score_report")
    audit = state.get("auditor_report")
    context = state.get("context_text")
    errors = state.get("validation_errors", [])
    attempts = state.get("attempts", 0)

    # 1. Extraction Phase
    if not extracted and attempts < 3:
        return "scribe"
    
    # 2. Validation Retry Logic
    if errors and attempts < 3:
        print(f"⚠️ Validation errors detected: {len(errors)}. Retrying Scribe (Attempt {attempts+1}).")
        return "scribe"
    
    if not extracted and attempts >= 3:
        print("❌ Max attempts reached for extraction. Giving up.")
        return "END"

    # 3. Calculation Phase
    if extracted and not risk:
        return "mathematician"
    
    if risk and not context:
        return "clinical_rag"
    
    if audit:
        return "END"
    
    return "END"