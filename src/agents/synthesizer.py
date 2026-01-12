from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from src.state.agent_state import AgentState

def synthesizer_node(state: AgentState) -> AgentState:
    print("--- 📝 NODE: SYNTHESIZER ---")
    # Pega tudo do estado
    patient_data = state.get("extracted_data")
    risk = state.get("risk_score_report", "N/A")
    audit_context = state.get("context_text", "N/A")

    system_prompt = ""
    user_prompt = f"""Here are the data from the previus nodes:
        {
            "patient_data": patient_data,
            "risk_report": risk,
            "auditor_report": audit_context
        }
    """
    system_message = SystemMessage(content=system_prompt)
    user_message = HumanMessage(content=user_prompt)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([system_prompt, user_message])
    state["final_report"] = response.content
    return state