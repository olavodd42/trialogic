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

SEED = 42

# Carrega o Prompt
prompt_path = os.path.join(os.getcwd(), "prompts", "auditor_prompt.md")
with open(prompt_path, encoding='utf-8') as f:
    AUDITOR_SYSTEM = f.read()

# --- NOVO: Carrega as Definições Estáticas (A Âncora de Verdade) ---
def load_static_definitions() -> str:
    """Carrega o definitions.txt para garantir que as regras estejam sempre no contexto."""
    def_path = os.path.join(os.getcwd(), "docs", "definitions.txt")
    try:
        with open(def_path, "r", encoding="utf-8") as f:
            return f"\n\n--- OFFICIAL PROTOCOL DEFINITIONS (ALWAYS USE THESE) ---\n{f.read()}\n----------------------------------------------------\n"
    except FileNotFoundError:
        print("⚠️ Warning: definitions.txt not found.")
        return ""

STATIC_RULES = load_static_definitions()
# -------------------------------------------------------------------

def check_quote_fidelity(quote: str, context: str, threshold=0.5) -> bool: # Mantive 0.5 que é mais seguro
    if "Not defined" in quote or "Retrieved context insufficient" in quote:
        return True
        
    quote_clean = " ".join(quote.lower().split())
    context_clean = " ".join(context.lower().split())
    
    if quote_clean in context_clean:
        return True
        
    match = SequenceMatcher(None, quote_clean, context_clean).find_longest_match(0, len(quote_clean), 0, len(context_clean))
    score = match.size / len(quote_clean) if len(quote_clean) > 0 else 0
    
    # Debug print para você ver o que está acontecendo
    print(f"🔍 Quote Fidelity Check: Score {score:.2f} (Threshold {threshold})")
    
    return score > threshold

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    print("--- ⚖️ NODE: SYNTHESIZER (AUDITOR) ---")

    extracted_data = state.get("extracted_data")
    risk_report = state.get("risk_score_report")
    rag_context = state.get("context_text", "")

    # --- A MÁGICA ACONTECE AQUI ---
    # Combinamos o que o RAG achou (dinâmico) com as Regras Fixas (estático)
    # Isso garante que a tabela do NEWS2 esteja SEMPRE presente.
    full_context_for_llm = f"{STATIC_RULES}\n\n--- RAG RETRIEVED CONTEXT ---\n{rag_context}"
    # ------------------------------

    full_patient_context = f"Data: {extracted_data}. Risk Assessment: {risk_report}"

    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key=SecretStr("lm-studio"),
        model="gpt-4o-mini", # Ou o modelo que estiver usando no LM Studio
        temperature=0,
        seed=SEED
    )
    structured_llm = llm.with_structured_output(AuditorEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", AUDITOR_SYSTEM),
        ("human", "Audit this case based on the retrieved context.")
    ])

    chain = prompt | structured_llm

    try:
        evaluation = cast(AuditorEvaluation, chain.invoke({
            "context": full_context_for_llm, # Passamos o contexto enriquecido
            "patient_state": full_patient_context
        }))

        # Validação de fidelidade agora vai funcionar porque o texto ESTÁ lá
        is_faithful = check_quote_fidelity(evaluation.evidence_quote, full_context_for_llm)
        
        if not is_faithful:
            print(f"🚨 HALLUCINATION CAUGHT: Quote '{evaluation.evidence_quote}' not found.")
            evaluation.compliance = "Inconclusive"
            evaluation.evidence_quote = "MODEL HALLUCINATION DETECTED: The model attempted to cite text not present in the source documents."
            evaluation.protocol_reference = "N/A"
        
        print(f"📝 Veredito: {evaluation.compliance}")

        return {
            "auditor_report": evaluation.model_dump()
        }

    except Exception as e:
        print(f"❌ Error in synthesis: {e}")
        return {"auditor_report": {"error": str(e), "compliance": "Inconclusive"}}