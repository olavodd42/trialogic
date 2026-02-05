
# from difflib import SequenceMatcher
# def check_quote_fidelity(quote: str, context: str, threshold=0.3) -> bool:
#     """
#     Verifies if the cited quote exists within the provided context using fuzzy matching.

#     Args:
#         quote (str): The evidence quote extracted by the LLM.
#         context (str): The full context text to search within.
#         threshold (float): Minimum similarity threshold for fuzzy matching.

#     Returns:
#         bool: True if the quote is consistent with the context, False otherwise.
#     """
#     # 1. Ignore validations in explicit error cases
#     if "Missing data" in quote or "not found" in quote.lower():
#         return True

#     # 2. Clean the data
#     quote_clean = " ".join(quote.lower().split())
#     context_clean = " ".join(context.lower().split())
    
#     # 3. Exact Match
#     if quote_clean in context_clean:
#         return True
        
#     # 4. Fuzzy Match
#     match = SequenceMatcher(None, quote_clean, context_clean).find_longest_match(0, len(quote_clean), 0, len(context_clean))
#     score = match.size / len(quote_clean) if len(quote_clean) > 0 else 0
    
#     if score > threshold:
#         return True

#     # 5. Keyword Rescue
#     keywords = ["sbp", "mmhg", "mews", "news", "score", "rate", "temp", "sepsis", "hypotension", "tachycardia"]
#     hits = sum(1 for k in keywords if k in quote_clean)

#     return hits >= 2

import re
import logging
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

def _normalize_text(s: str) -> str:
    s = s or ""
    # preserva números, % e / (para BP 120/80), remove pontuação "decorativa"
    s = s.lower()
    s = re.sub(r"[“”\"'«»`]", "", s)
    s = re.sub(r"[\(\)\[\]\{\}:;,\—\–\*]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _sentences(text: str):
    # split simples por sentenças — suficiente para nossa análise
    text = text or ""
    parts = re.split(r'(?<=[\.\?\!\n])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def _token_set(s: str):
    return set(re.findall(r"\b[a-z0-9%/]{2,}\b", s.lower()))

def _bigrams(s: str):
    toks = list(re.findall(r"\b[a-z0-9%/]{2,}\b", s.lower()))
    return set(zip(toks, toks[1:])) if len(toks) >= 2 else set()

def _extract_numbers(s: str):
    # encontra números, porcentagens e BPs (ex: 120/80)
    nums = re.findall(r"\d{1,3}(?:/\d{1,3})?(?:\.\d+)?%?", s)
    return nums

def check_quote_fidelity(quote: str, context: str, 
                        ratio_threshold_high: float = 0.85,
                        ratio_threshold_medium: float = 0.60,
                        jaccard_threshold: float = 0.40,
                        bigram_threshold: float = 0.35,
                        numeric_strict: bool = True) -> bool:
    """
    Verifica se `quote` está adequado ao `context` usando várias heurísticas:
      1) correspondência exata,
      2) comparação por sentença com SequenceMatcher (ratios altos = match forte),
      3) Jaccard token-level e bigram overlap,
      4) correspondência numérica (se quote contém números, estes aumentam confiança).

    Retorna True se considerarmos a citação fiel ao contexto.

    Parâmetros de sensibilidade podem ser ajustados pelos argumentos.
    """

    # 0. Casos explícitos (mantidos compatíveis com a função antiga)
    if not quote:
        return False  # quote vazio -> não fiel (caller pode optar por fallback)
    if "missing data" in quote.lower() or "not found" in quote.lower():
        return True

    quote_clean = _normalize_text(quote)
    context_clean = _normalize_text(context)

    # 1. Exact substring match (mais forte)
    if quote_clean in context_clean:
        logger.debug("check_quote_fidelity: exact substring match")
        return True

    # 2. If quote very short, require substring only (evita falso-positivo)
    if len(quote_clean) < 20:
        # para citações curtas, aceitar se tokens-chave >=2 presentes no contexto
        q_tokens = _token_set(quote_clean)
        c_tokens = _token_set(context_clean)
        hits = len(q_tokens & c_tokens)
        logger.debug(f"short quote tokens hits={hits}")
        return hits >= 2

    # 3. Sentence-level SequenceMatcher: calcula o melhor ratio entre quote e cada sentença do contexto
    sentences = _sentences(context_clean)
    best_ratio = 0.0
    best_sentence = ""
    for s in sentences:
        if not s:
            continue
        r = SequenceMatcher(None, quote_clean, s).ratio()
        if r > best_ratio:
            best_ratio = r
            best_sentence = s
    logger.debug(f"best_ratio={best_ratio:.3f} best_sentence_preview={best_sentence[:120]}")

    if best_ratio >= ratio_threshold_high:
        logger.debug("check_quote_fidelity: high ratio match")
        return True

    # 4. Token-level Jaccard + bigram overlap with the best candidate sentence
    quote_tokens = _token_set(quote_clean)
    sent_tokens = _token_set(best_sentence)
    jaccard = 0.0
    if quote_tokens or sent_tokens:
        jaccard = len(quote_tokens & sent_tokens) / max(1, len(quote_tokens | sent_tokens))
    bigr_q = _bigrams(quote_clean)
    bigr_s = _bigrams(best_sentence)
    bigram_overlap = 0.0
    if bigr_q or bigr_s:
        bigram_overlap = len(bigr_q & bigr_s) / max(1, len(bigr_q | bigr_s))

    logger.debug(f"jaccard={jaccard:.3f} bigram_overlap={bigram_overlap:.3f}")

    # 5. Numeric evidence: se a quote inclui números (ex: 95%, 120/80), exijimos correspondência numérica forte
    q_nums = _extract_numbers(quote_clean)
    s_nums = _extract_numbers(best_sentence)
    numeric_match = False
    if q_nums:
        # se todos os números do quote aparecem no sentence -> match forte
        numeric_match = all(any(qn == sn for sn in s_nums) for qn in q_nums)
        logger.debug(f"numeric_match={numeric_match} q_nums={q_nums} s_nums={s_nums}")

        if numeric_strict and numeric_match:
            return True
        # se números presentes mas não batem, reduzir confiança
        if numeric_strict and not numeric_match:
            return False

    # 6. Heurística combinada:
    # - medium ratio + jaccard or bigram suficiente -> accept
    if best_ratio >= ratio_threshold_medium and (jaccard >= jaccard_threshold or bigram_overlap >= bigram_threshold):
        logger.debug("check_quote_fidelity: medium ratio + lexical overlap -> accept")
        return True

    # - jaccard + bigram strong
    if (jaccard >= max(0.5, jaccard_threshold)) and (bigram_overlap >= max(0.45, bigram_threshold)):
        logger.debug("check_quote_fidelity: strong token/bigram overlap -> accept")
        return True

    # fallback pelo número de keywords clínicos compartilhados (rescue da função antiga)
    keywords = {"sbp", "mmhg", "mews", "news", "score", "rate", "temp", "sepsis", "hypotension", "tachycardia", "pneumonia", "oxygen", "respiratory"}
    hits = sum(1 for k in keywords if k in quote_clean and k in context_clean)
    logger.debug(f"keyword_hits={hits}")
    if hits >= 2:
        return True

    # caso contrário, não fiel
    return False
