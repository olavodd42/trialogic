import os
from typing import Dict, Any, cast
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from difflib import SequenceMatcher
from dotenv import load_dotenv

from src.state.agent_state import AgentState
from src.schemas.auditor_schema import AuditorEvaluation

load_dotenv()

# Load prompt with cross-platform path handling
prompt_path = os.path.join(os.getcwd(), "prompts", "auditor_prompt.md")
with open(prompt_path, encoding='utf-8') as f:
    AUDITOR_SYSTEM = f.read()

def check_quote_fidelity(quote: str, context: str, threshold=0.8) -> bool:
    """Verifica se a citação realmente existe no contexto (Fuzzy match)."""
    if "Not defined" in quote or "Retrieved context insufficient" in quote:
        return True
        
    # Normalização simples
    quote_clean = " ".join(quote.lower().split())
    context_clean = " ".join(context.lower().split())
    
    if quote_clean in context_clean:
        return True
        
    # Se não for exato, tenta fuzzy (para lidar com pontuação/espaços)
    match = SequenceMatcher(None, quote_clean, context_clean).find_longest_match(0, len(quote_clean), 0, len(context_clean))
    similarity = match.size / len(quote_clean)
    print(f"🔍 Quote Fidelity Score: {similarity:.2f} (Threshold: {threshold})")
    
    return similarity > threshold

# --- NODE LOGIC ---
def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Orchestrates the Synthesis (Auditor) process.

    This node acts as the auditor in the workflow, evaluating the patient's case against
    retrieved clinical guidelines. It uses a structured LLM to generate a compliance report.

    Args:
        state (AgentState): The current state of the agent workflow, containing:
            - extracted_data: The structured patient data.
            - risk_score_report: The risk assessment report.
            - context_text: The retrieved clinical guidelines (from RAG).

    Returns:
        Dict[str, Any]: A dictionary containing the auditor's report:
            - auditor_report (AuditorEvaluation): A structured evaluation of compliance.
    """
    print("--- ⚖️ NODE: SYNTHESIZER (AUDITOR) ---")

    # Recover data from previous nodes
    extracted_data = state.get("extracted_data")
    risk_report = state.get("risk_score_report")
    retrieved_context = state.get("context_text", "No context provided.")

    full_patient_context = f"Data: {extracted_data}. Risk Assessment: {risk_report}"

    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key=SecretStr("lm-studio"),
        model="gpt-4o-mini",
        temperature=0
    )
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

        is_faithful = check_quote_fidelity(evaluation.evidence_quote, retrieved_context or "")
        
        if not is_faithful:
            print(f"🚨 HALLUCINATION CAUGHT: Quote '{evaluation.evidence_quote}' not found in context.")
            # Penaliza o modelo ou marca como inconclusivo
            evaluation.compliance = "Inconclusive"
            evaluation.evidence_quote = "MODEL HALLUCINATION DETECTED: The model attempted to cite text not present in the source documents."
            evaluation.protocol_reference = "N/A"
        
        print(prompt)
        print(f"📝 Veredito: {evaluation.compliance}")

        return {
            "auditor_report": evaluation.model_dump()
        }
    
    except Exception as e:
        print(f"❌ Error in synthesis: {e}")
        return {"auditor_report": {"error": str(e), "compliance": "Inconclusive"}}