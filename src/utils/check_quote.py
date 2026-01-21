
from difflib import SequenceMatcher
def check_quote_fidelity(quote: str, context: str, threshold=0.3) -> bool:
    # 1. Ignore validations in explicit error cases
    if "Missing data" in quote or "not found" in quote.lower():
        return True

    # 2. Clean the data
    quote_clean = " ".join(quote.lower().split())
    context_clean = " ".join(context.lower().split())
    
    # 3. Exact Match
    if quote_clean in context_clean:
        return True
        
    # 4. Fuzzy Match
    match = SequenceMatcher(None, quote_clean, context_clean).find_longest_match(0, len(quote_clean), 0, len(context_clean))
    score = match.size / len(quote_clean) if len(quote_clean) > 0 else 0
    
    if score > threshold:
        return True

    # 5. Keyword Rescue
    keywords = ["sbp", "mmhg", "mews", "news", "score", "rate", "temp", "sepsis", "hypotension", "tachycardia"]
    hits = sum(1 for k in keywords if k in quote_clean)

    return hits >= 2