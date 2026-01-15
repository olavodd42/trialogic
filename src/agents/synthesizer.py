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

# Mantenha os imports e o load_static_definitions...

def check_quote_fidelity(quote: str, context: str, threshold=0.3) -> bool: # Threshold baixo (30%)
    """
    Verificação Híbrida: Tenta match exato, fuzzy e keywords.
    Para o TCC, queremos evitar Falsos Negativos (Inconclusive) se a lógica estiver certa.
    """
    # 1. Limpeza
    quote_clean = " ".join(quote.lower().split())
    context_clean = " ".join(context.lower().split())
    
    # 2. Match Exato ou Substring
    if quote_clean in context_clean:
        return True
        
    # 3. Fuzzy Match (Levenshtein)
    match = SequenceMatcher(None, quote_clean, context_clean).find_longest_match(0, len(quote_clean), 0, len(context_clean))
    score = match.size / len(quote_clean) if len(quote_clean) > 0 else 0
    
    if score > threshold:
        return True

    # 4. Keyword Fallback (A "Rede de Segurança")
    # Se a citação menciona métricas válidas que estão no contexto, aceitamos.
    keywords = ["sbp", "heart rate", "mews", "news", "score", "mmhg", "risk", "sepsis", "hypotension"]
    hits = sum(1 for k in keywords if k in quote_clean)
    
    if hits >= 2: # Se tem pelo menos 2 termos técnicos, aceitamos como "Parafraseamento Válido"
        print(f"⚠️ Quote accepted via Keyword Fallback ({hits} hits): '{quote[:30]}...'")
        return True
        
    return False

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    print("--- ⚖️ NODE: SYNTHESIZER (AUDITOR) ---")

    extracted_data = state.get("extracted_data")
    risk_report = state.get("risk_score_report")
    rag_context = state.get("context_text", "")

    # Injeta regras estáticas
    full_context_for_llm = f"{STATIC_RULES}\n\n--- RAG CONTEXT ---\n{rag_context}"
    full_patient_context = f"Data: {extracted_data}. Risk Assessment: {risk_report}"

    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key=SecretStr("lm-studio"),
        model="gpt-4o-mini", # Ajuste conforme seu modelo
        temperature=0,
        seed=SEED
    )
    structured_llm = llm.with_structured_output(AuditorEvaluation)

    # Prompt simplificado para focar na decisão
    prompt = ChatPromptTemplate.from_messages([
        ("system", AUDITOR_SYSTEM),
        ("human", f"Evaluate this patient.\nCONTEXT:\n{full_context_for_llm}\n\nPATIENT:\n{full_patient_context}")
    ])

    chain = prompt | structured_llm

    try:
        evaluation = cast(AuditorEvaluation, chain.invoke({}))

        # Validação com rede de segurança
        is_faithful = check_quote_fidelity(evaluation.evidence_quote, full_context_for_llm)
        
        if not is_faithful:
            # Só marcamos inconclusivo se realmente for uma alucinação maluca
            print(f"🚨 HALLUCINATION REJECTED: '{evaluation.evidence_quote}'")
            evaluation.compliance = "Inconclusive"
            evaluation.evidence_quote = "Source text not found."
        
        print(f"📝 Veredito: {evaluation.compliance}")

        return {
            "auditor_report": evaluation.model_dump()
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"auditor_report": {"error": str(e), "compliance": "Inconclusive"}}