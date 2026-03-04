"""Quote-fidelity checker for RAG evidence citations.

Provides heuristic-based verification that an LLM-generated quote
faithfully reflects the retrieved context, using exact matching,
SequenceMatcher ratios, Jaccard overlap, bigram overlap, and
numeric-evidence matching.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

def _normalize_text(s: str) -> str:
    """Lower-case, strip quotes/brackets/punctuation and collapse whitespace."""
    s = s or ""
    s = s.lower()
    s = re.sub(r"[“”\"'«»`]", "", s)
    s = re.sub(r"[\(\)\[\]\{\}:;,\—\–\*]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _sentences(text: str) -> List[str]:
    """Split *text* into sentences on '.', '?', '!', or newline boundaries."""
    text = text or ""
    parts = re.split(r'(?<=[\.\?\!\n])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def _token_set(s: str) -> Set[str]:
    """Return the set of alphanumeric tokens (length >= 2) in *s*."""
    return set(re.findall(r"\b[a-z0-9%/]{2,}\b", s.lower()))

def _bigrams(s: str) -> Set[Tuple[str, str]]:
    """Return the set of consecutive-token bigrams in *s*."""
    toks = list(re.findall(r"\b[a-z0-9%/]{2,}\b", s.lower()))
    return set(zip(toks, toks[1:])) if len(toks) >= 2 else set()

def _extract_numbers(s: str) -> List[str]:
    """Extract numbers, percentages, and BP-style values (e.g. 120/80)."""
    nums = re.findall(r"\d{1,3}(?:/\d{1,3})?(?:\.\d+)?%?", s)
    return nums

def check_quote_fidelity(quote: str, context: str, 
                        ratio_threshold_high: float = 0.85,
                        ratio_threshold_medium: float = 0.60,
                        jaccard_threshold: float = 0.40,
                        bigram_threshold: float = 0.35,
                        numeric_strict: bool = True) -> bool:
    """
    Verify whether *quote* faithfully represents *context* using several heuristics.

    Checks applied (in order):
      1. Exact substring match.
      2. Per-sentence SequenceMatcher ratio (high ratio = strong match).
      3. Jaccard token-level and bigram overlap.
      4. Numeric evidence matching (numbers in the quote must appear in context).

    Returns ``True`` when the citation is considered faithful.
    Sensitivity parameters can be tuned via the keyword arguments.
    """

    # 0. Explicit edge cases (kept compatible with the legacy function)
    if not quote:
        return False  # Empty quote -> not faithful (caller may use a fallback)
    if "missing data" in quote.lower() or "not found" in quote.lower():
        return True

    quote_clean = _normalize_text(quote)
    context_clean = _normalize_text(context)

    # 1. Exact substring match (strongest signal)
    if quote_clean in context_clean:
        logger.debug("check_quote_fidelity: exact substring match")
        return True

    # 2. If quote is very short, require substring only (avoids false positives)
    if len(quote_clean) < 20:
        # For short quotes, accept if >= 2 key tokens are present in the context
        q_tokens = _token_set(quote_clean)
        c_tokens = _token_set(context_clean)
        hits = len(q_tokens & c_tokens)
        logger.debug("Short quote token hits=%d", hits)
        return hits >= 2

    # 3. Sentence-level SequenceMatcher: compute the best ratio between quote and each context sentence
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
    logger.debug("best_ratio=%.3f best_sentence_preview=%s", best_ratio, best_sentence[:120])

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

    logger.debug("jaccard=%.3f bigram_overlap=%.3f", jaccard, bigram_overlap)

    # 5. Numeric evidence: if the quote includes numbers (e.g. 95%, 120/80), require strong numeric matching
    q_nums = _extract_numbers(quote_clean)
    s_nums = _extract_numbers(best_sentence)
    numeric_match = False
    if q_nums:
        # If all numbers in the quote appear in the sentence -> strong match
        numeric_match = all(any(qn == sn for sn in s_nums) for qn in q_nums)
        logger.debug("numeric_match=%s q_nums=%s s_nums=%s", numeric_match, q_nums, s_nums)

        if numeric_strict and numeric_match:
            return True
        # Numbers present but don't match -> reduce confidence
        if numeric_strict and not numeric_match:
            return False

    # 6. Combined heuristic:
    # - medium ratio + sufficient jaccard or bigram -> accept
    if best_ratio >= ratio_threshold_medium\
        and (jaccard >= jaccard_threshold or bigram_overlap >= bigram_threshold):
        logger.debug("check_quote_fidelity: medium ratio + lexical overlap -> accept")
        return True

    # - strong jaccard + bigram
    if (jaccard >= max(0.5, jaccard_threshold))\
        and (bigram_overlap >= max(0.45, bigram_threshold)):
        logger.debug("check_quote_fidelity: strong token/bigram overlap -> accept")
        return True

    # Fallback: shared clinical keywords (rescue from the legacy function)
    keywords = {
        "sbp", "mmhg", "mews", "news", "score", "rate",
        "temp", "sepsis", "hypotension", "tachycardia",
        "pneumonia", "oxygen", "respiratory"
    }
    hits = sum(1 for k in keywords if k in quote_clean and k in context_clean)
    logger.debug("keyword_hits=%d", hits)
    if hits >= 2:
        return True

    # Otherwise, not faithful
    return False
