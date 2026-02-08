from typing import List, Literal, Dict
from src.state.agent_state import AgentState
import logging

logger = logging.getLogger(__name__)

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
    return {"plan": initial_plan} 

def supervisor_router(state: AgentState) -> Literal["scribe", "mathematician", "clinical_rag", "END"]:
    """
    Supervisor Router: Determines the next step in the workflow based on the state.

    Logic:
    1. If no extracted data, route to 'scribe'.
    2. If validation errors exist and attempts < max, retry 'scribe'.
    3. If structured data exists but no risk score, route to 'mathematician'.
    4. If risk scores exist but no contextual grounding, route to 'clinical_rag'.
    5. If all steps are complete, terminate execution.
    """
    
    extracted = state.get("extracted_data")
    risk = state.get("risk_score_report")
    audit = state.get("auditor_report")
    context = state.get("context_text")
    errors = state.get("validation_errors", [])
    attempts = state.get("attempts", 0)

    # 1. Extraction Phase
    if not extracted and attempts < 3:
        # logging.info("DEBUG AGENT: Routing to scribe (No extraction)")
        return "scribe"
    
    # 2. Validation Retry Logic
    if errors and attempts < 3:
        logging.warning(f"⚠️ Validation errors detected: {len(errors)}. Retrying Scribe (Attempt {attempts+1}).")
        return "scribe"
    
    if not extracted and attempts >= 3:
        logging.error("❌ Max attempts reached for extraction. Giving up.")
        return "END"

    # 3. Calculation Phase
    if extracted and not risk:
        # print("DEBUG AGENT: Routing to mathematician")
        return "mathematician"
    
    if risk and not context:
        # print("DEBUG AGENT: Routing to clinical_rag")
        return "clinical_rag"
    
    if audit:
        # print("DEBUG AGENT: Routing to END (Audit complete)")
        return "END"
    
    # print("DEBUG AGENT: Routing to END (Default)")
    return "END"