import os
from typing import Dict, Any, cast
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.state.agent_state import AgentState
from src.schemas.auditor_schema import AuditorEvaluation

load_dotenv()

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

with open(os.path.join(os.getcwd().replace("\\", "/"), "prompts/auditor_prompt.md")) as f:
    AUDITOR_SYSTEM = f.read()

# --- NODE LOGIC ---
def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    print("--- ⚖️ NODE: SYNTHESIZER (AUDITOR) ---")

    # Recovery data from previous nodes
    extracted_data = state.get("extracted_data")
    risk_report = state.get("risk_score_report")
    retrieved_context = state.get("context_text", "No context provided.")

    full_patient_context = f"Data: {extracted_data}. Risk Assessment: {risk_report}"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(AuditorEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", AUDITOR_SYSTEM),
        ("human", "Audit this case based on the retrieved context.")
    ])

    chain = prompt | structured_llm

    try:
        evaluation = cast(AuditorEvaluation, chain.invoke({
            "context": retrieved_context,
            "patient_state": full_patient_context
        }))
        
        print(f"📝 Veredito: {evaluation.compliance}")

        return {
            "auditor_report": evaluation.model_dump()
        }
    
    except Exception as e:
        print(f"❌ Error in synthesis: {e}")
        return {"auditor_report": {"error": str(e), "compliance": "Inconclusive"}}