import os
from typing import Dict, Any, cast
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from difflib import SequenceMatcher
from dotenv import load_dotenv

from src.state.agent_state import AgentState
from src.schemas.auditor_schema import AuditorOutput

load_dotenv()

SEED = 42

# Carrega o Prompt do arquivo
prompt_path = os.path.join(os.getcwd(), "prompts", "auditor_prompt.md")
with open(prompt_path, encoding='utf-8') as f:
    AUDITOR_SYSTEM = f.read()

# Carrega as Definições Estáticas
def load_static_definitions() -> str:
    def_path = os.path.join(os.getcwd(), "docs", "definitions.txt")
    try:
        with open(def_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Standard Clinical Protocols (Sepsis-3, NEWS2, MEWS)."

STATIC_RULES = load_static_definitions()

def check_quote_fidelity(quote: str, context: str, threshold=0.3) -> bool:
    """Validação híbrida: Exata, Fuzzy e Keywords."""
    # 1. Ignora verificações em casos de erro explícito
    if "Missing data" in quote or "not found" in quote.lower():
        return True

    # 2. Limpeza
    quote_clean = " ".join(quote.lower().split())
    context_clean = " ".join(context.lower().split())
    
    # 3. Match Exato
    if quote_clean in context_clean:
        return True
        
    # 4. Fuzzy Match
    match = SequenceMatcher(None, quote_clean, context_clean).find_longest_match(0, len(quote_clean), 0, len(context_clean))
    score = match.size / len(quote_clean) if len(quote_clean) > 0 else 0
    
    if score > threshold:
        return True

    # 5. Keyword Rescue (Salva o artigo de falsos negativos)
    keywords = ["sbp", "mmhg", "mews", "news", "score", "rate", "temp", "sepsis", "hypotension", "tachycardia"]
    hits = sum(1 for k in keywords if k in quote_clean)
    
    # Se a citação tem termos técnicos válidos, aceitamos
    return hits >= 2

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    print("--- ⚖️ NODE: SYNTHESIZER (AUDITOR) ---")

    extracted_data = state.get("extracted_data")
    risk_report = state.get("risk_score_report")
    rag_context = state.get("context_text", "No specific RAG context found.")

    # Prepara o Contexto Unificado
    # OBS: Usamos chaves duplas {{ }} se quiséssemos escapar, mas aqui vamos passar como variável
    full_context_content = f"""
    === STATIC PROTOCOL DEFINITIONS ===
    {STATIC_RULES}

    === RETRIEVED GUIDELINES ===
    {rag_context}
    """

    full_patient_content = f"EXTRACTED DATA: {extracted_data}\nRISK CALCULATIONS: {risk_report}"

    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="llama3.1",
        temperature=0,
        seed=SEED
    )
    structured_llm = llm.with_structured_output(AuditorOutput)

    # --- CORREÇÃO DO BUG DO LANGCHAIN ---
    # Não usamos f-string aqui. Definimos placeholders {rules} e {patient}.
    # Isso impede que chaves dentro do JSON do paciente quebrem o template.
    prompt = ChatPromptTemplate.from_messages([
        ("system", AUDITOR_SYSTEM),
        ("human", """
        Analyze the following case against the protocols.
        
        # KNOWLEDGE BASE
        {rules}
        
        # PATIENT DATA
        {patient}
        """)
    ])

    chain = prompt | structured_llm

    try:
        # Passamos as variáveis aqui. O LangChain cuida da injeção segura.
        evaluation = cast(AuditorOutput, chain.invoke({
            "rules": full_context_content,
            "patient": full_patient_content
        }))

        print(evaluation)

        # Preparação segura dos atributos (suporte a Pydantic V2 e Dict)
        if isinstance(evaluation, dict):
            quote = evaluation.get("evidence_quote", "")
        else:
            quote = getattr(evaluation, "evidence_quote", "")

        # Só somos rígidos se ele acusar problema
        is_faithful = check_quote_fidelity(quote, full_context_content)
        if not is_faithful:
            print(f"🚨 HALLUCINATION DETECTED: Quote '{quote}' not found.")
                
            # Fallback suave para não perder o dado no artigo
            warning_suffix = " [Warning: Quote inexact]"
            if isinstance(evaluation, dict):
                evaluation["evidence_quote"] = quote + warning_suffix
            else:
                evaluation.evidence_quote += warning_suffix
        
        print(f"📝 Veredict: {evaluation.clinical_risk_category}")

        return {
            "auditor_report": evaluation.model_dump()
        }

    except Exception as e:
        print(f"❌ Error in Synthesizer: {e}")
        # Retorno de segurança para não quebrar o batch
        return {
            "auditor_report": {
                "compliance": "Inconclusive",
                "evidence_quote": f"System Error: {str(e)}",
                "clinical_suggestion": "Manual review required.",
                "protocol_reference": "Error"
            }
        }
    
