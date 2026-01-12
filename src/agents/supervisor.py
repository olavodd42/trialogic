from src.state.agent_state import AgentState

def supervisor_planning(state: AgentState) -> AgentState:

    print("--- 🧠 NODE: SUPERVISOR (PLANNING) ---")

    initial_plan = ["scribe", "mathematician", "auditor"]
    state["plan"] = initial_plan
    return state

def supervisor_router(state: AgentState) -> str:
    extracted = state.get("extracted_data")
    risk = state.get("risk_score_report")
    audit = state.get("auditor_report")

    if not extracted:
        return "scribe"
    
    if extracted and not risk:
        return "mathematician"
    
    if risk and not audit:
        return "auditor"
    
    return "END"