import os
import re
from difflib import SequenceMatcher
from typing import Dict, Tuple, Optional, Any, cast
from langchain_openai import ChatOpenAI
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

# def load_static_definitions() -> str:
#     """
#     Loads static clinical protocol definitions from a local text file.

#     Returns:
#         str: The content of the definitions file. Returns a default string 
#              referencing standard protocols (Sepsis-3, NEWS2, MEWS) if the file is not found.
#     """
#     def_path = os.path.join(os.getcwd(), "docs", "definitions.txt")
#     try:
#         with open(def_path, "r", encoding="utf-8") as f:
#             return f.read()
#     except FileNotFoundError:
#         return "Standard Clinical Protocols (Sepsis-3, NEWS2, MEWS)."

# STATIC_RULES = load_static_definitions()

def _split_docs(rag_context: str):
    """
    Espera que retrieved_context tenha blocos formatados como:
    SOURCE [path]: content...
    Retorna lista de (source, text).
    """
    blocks = []
    if not rag_context:
        return blocks
    # tenta dividir por "SOURCE [" que você já gera no RAG
    parts = re.split(r'(SOURCE \[.*?\]:)', rag_context)
    # parts comes in alternating markers/content; rebuild pairs
    cur_source = None
    cur_text = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if part.startswith("SOURCE ["):
            # next part is content
            source = part
            i += 1
            content = parts[i] if i < len(parts) else ""
            blocks.append((source, content.strip()))
        i += 1
    # fallback: if no SOURCE markers, return whole as one doc
    if not blocks:
        blocks = [("RAG", rag_context)]
    return blocks

def _sentences_of(text: str):
    # split simples em sentenças
    s = re.split(r'(?<=[\.\?\!\n])\s+', text)
    return [x.strip() for x in s if x.strip()]

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def extract_best_evidence_from_context(rag_context: str, 
                                       target_texts: list,    # lista de strings para comparar (reasoning, suggestion, etc)
                                       min_ratio: float = 0.45
                                      ) -> Tuple[Optional[str], Optional[str], float]:
    """
    Busca a melhor sentença no rag_context que suporte os target_texts.
    Retorna (best_sentence, source_label, best_score). Se none, retorna (None, None, 0.0).
    """
    if not rag_context:
        return None, None, 0.0

    docs = _split_docs(rag_context)
    best = (None, None, 0.0)  # sentence, source, score

    # normalize targets
    targets = [(" ".join(t.lower().split())) for t in target_texts if t]
    if not targets:
        return None, None, 0.0

    for source, content in docs:
        sentences = _sentences_of(content)
        for sent in sentences:
            sent_norm = " ".join(sent.lower().split())
            # compute best similarity across all target_texts (max)
            scores = [_similarity(sent_norm, tgt) for tgt in targets]
            score = max(scores) if scores else 0.0
            # boost if share numeric tokens (e.g., 95%, 120/80)
            nums_in_sent = re.findall(r"\d{1,3}(?:/\d{1,3})?(?:\.\d+)?%?", sent_norm)
            nums_in_targets = []
            for t in targets:
                nums_in_targets += re.findall(r"\d{1,3}(?:/\d{1,3})?(?:\.\d+)?%?", t)
            num_boost = 0.15 if any(n in nums_in_sent for n in nums_in_targets) and nums_in_targets else 0.0
            score += num_boost
            if score > best[2]:
                best = (sent.strip(), source.strip(), score)

    # retorna somente se passar threshold razoável
    if best[2] >= min_ratio:
        return best  # sentence, source, score
    return None, None, 0.0

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Synthesizer (Auditor) node logic.

    This function aggregates extracted patient data, risk score reports, and 
    retrieved RAG context to generate a comprehensive clinical audit using an LLM.
    It constructs the context, prompts the model for a structured evaluation, 
    and verifies the fidelity of cited evidence against the source context.

    Args:
        state (AgentState): The current state of the agent workflow, containing key 
                            keys like 'extracted_data', 'risk_score_report', and 'context_text'.

    Returns:
        Dict[str, Any]: A dictionary containing the 'auditor_report' to update the state. 
                        In case of error, returns an error report.
    """
    logger.info("--- ⚖️ NODE: SYNTHESIZER (AUDITOR) ---")

    extracted_data = state.get("extracted_data")
    risk_report = state.get("risk_score_report")
    rag_context = state.get("rag_context", "")
    rag_context_used = state.get("rag_context_used", False)

    logger.debug(f"RAG_CONTEXT LEN: {len(rag_context) if rag_context else 0}")
    logger.debug(f"RAG_CONTEXT PREVIEW:\n{rag_context[:500] if rag_context else ""}")

    # 1. Describe o contexto condicionalmente
    if True:
        full_context_content = f"""
        === RETRIEVED CLINICAL GUIDELINES ===
        {rag_context}
        """
    else:
        full_context_content = ""
    
    full_patient_content = f"EXTRACTED DATA: {extracted_data}\nRISK CALCULATIONS: {risk_report}"

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        seed=42
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

    logger.debug(f"PROMPT: {prompt}")

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

            fallback_targets = []
            try:
                # evaluation pode ser dict ou objeto
                cs = evaluation.get("clinical_suggestion") if isinstance(evaluation, dict) else getattr(evaluation, "clinical_suggestion", "")
                rt = evaluation.get("reasoning_trace") if isinstance(evaluation, dict) else getattr(evaluation, "reasoning_trace", "")
                fallback_targets = [cs or "", rt or ""]
            except Exception:
                fallback_targets = []
                
            found_quote, found_source, found_score = extract_best_evidence_from_context(full_context_content, fallback_targets, min_ratio=0.45)
            if found_quote:
                logger.info(f"🔎 Fallback evidence found (score={found_score:.2f}) from {found_source}")
                # substitui evidence_quote pelo trecho exato encontrado (verbatim)
                if isinstance(evaluation, dict):
                    evaluation["evidence_quote"] = found_quote
                    evaluation["protocol_reference"] = found_source
                else:
                    evaluation.evidence_quote = found_quote
                    evaluation.protocol_reference = found_source or ""
            else:
                # se nada encontrado, acrescenta sufixo de warning como você já faz
                warning_suffix = " [Warning: Quote inexact]"
                if isinstance(evaluation, dict):
                    evaluation["evidence_quote"] = (quote or "") + warning_suffix
                else:
                    evaluation.evidence_quote = (quote or "") + warning_suffix

        logger.info(f"📝 Veredict: {evaluation.clinical_risk_category}")

        # Ajusta o campo protocol_reference se não houver contexto RAG
        auditor_report = evaluation.model_dump()
        if not rag_context_used:
            auditor_report["protocol_reference"] = ""
        return {
            "auditor_report": auditor_report
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
    
