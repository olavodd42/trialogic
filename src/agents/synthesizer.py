import os
from typing import Dict, Any, cast
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import logging
from dotenv import load_dotenv

from src.state.agent_state import AgentState
from src.schemas.auditor_schema import AuditorOutput
from src.utils.check_quote import check_quote_fidelity
load_dotenv()

SEED = 42
logger = logging.getLogger(__name__)
prompt_path = os.path.join(os.getcwd(), "prompts", "auditor_prompt.md")
with open(prompt_path, encoding='utf-8') as f:
    AUDITOR_SYSTEM = f.read()

def load_static_definitions() -> str:
    def_path = os.path.join(os.getcwd(), "docs", "definitions.txt")
    try:
        with open(def_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Standard Clinical Protocols (Sepsis-3, NEWS2, MEWS)."

STATIC_RULES = load_static_definitions()



def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- ⚖️ NODE: SYNTHESIZER (AUDITOR) ---")

    extracted_data = state.get("extracted_data")
    risk_report = state.get("risk_score_report")
    rag_context = state.get("context_text", "No specific RAG context found.")

    # 1. Describe the context
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
        logger.debug("Generating Evaluation via LLM...")
        # 2. Call the LLM to audit
        evaluation = cast(AuditorOutput, chain.invoke({
            "rules": full_context_content,
            "patient": full_patient_content
        }))

        # print(evaluation)

        if isinstance(evaluation, dict):
            quote = evaluation.get("evidence_quote", "")
        else:
            quote = getattr(evaluation, "evidence_quote", "")

        # 3. Check the evidence quote fidelity to RAG data
        is_faithful = check_quote_fidelity(quote, full_context_content)
        if not is_faithful:
            logger.warning(f"🚨 HALLUCINATION DETECTED: Quote '{quote}' not found.")
                
            warning_suffix = " [Warning: Quote inexact]"
            if isinstance(evaluation, dict):
                evaluation["evidence_quote"] = quote + warning_suffix
            else:
                evaluation.evidence_quote += warning_suffix
        
        logger.info(f"📝 Veredict: {evaluation.clinical_risk_category}")

        return {
            "auditor_report": evaluation.model_dump()
        }

    except Exception as e:
        logger.error(f"❌ Error in Synthesizer: {e}")
        return {
            "auditor_report": {
                "compliance": "Inconclusive",
                "evidence_quote": f"System Error: {str(e)}",
                "clinical_suggestion": "Manual review required.",
                "protocol_reference": "Error"
            }
        }
    
