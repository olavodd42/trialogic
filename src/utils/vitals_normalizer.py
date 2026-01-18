import re
from typing import Optional

def extract_number(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group(0)) if match else None

def normalize_temperature(raw: Optional[str]) -> Optional[float]:
    """
    Normaliza temperatura para Celsius.
    Regras:
    - Se contiver 'F' ou valor > 45 → assume Fahrenheit
    - Caso contrário → assume Celsius
    """
    if not raw:
        return None

    raw = raw.strip()
    value = extract_number(raw)
    if value is None:
        return None

    if "F" in raw.upper() or value > 45:
        return round((value - 32) * 5 / 9, 1)

    return round(value, 1)

# def normalize_vitals(raw: Optional[str]) -> Optional[float]:
#     """
#     Normaliza temperatura para Celsius.
#     Regras:
#     - Se contiver 'F' ou valor > 45 → assume Fahrenheit
#     - Caso contrário → assume Celsius
#     """
#     if not raw:
#         return None

#     raw = raw.strip()
#     value = extract_number(raw)
#     if value is None:
#         return None

#     if "F" in raw.upper() or value > 45:
#         return round((value - 32) * 5 / 9, 1)

#     return round(value, 1)