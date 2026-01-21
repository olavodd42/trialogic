import re
from typing import Optional, Union

def extract_number(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group(0)) if match else None

def normalize_temperature(raw: Union[str, float, int, None]) -> Optional[float]:
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        val = float(raw)
        if val > 45:
             val = (val - 32) * 5.0 / 9.0
        return round(val, 1)

    raw = str(raw).strip()
    value = extract_number(raw)
    if value is None:
        return None

    if "F" in raw.upper() or value > 45:
        return round((value - 32) * 5 / 9, 1)

    return round(value, 1)
