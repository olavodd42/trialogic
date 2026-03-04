"""Adapter utilities for normalising heterogeneous vital-sign structures.

Converts both 'rich' (nested) and 'flat' JSON formats coming from the
Scribe extraction into a uniform flat dictionary consumed by the
ClinicalCalculator tool.
"""

from typing import Any, Dict, List, Optional


def normalize_vitals_for_calculator(extraction_json: dict) -> dict:
    """
    Robust polymorphic adapter.

    Accepts both 'Rich JSON' (nested) and 'Flat JSON' (simple) formats
    and returns a flat dictionary ready for the ClinicalCalculator tool.
    """

    # 1. Try to locate where the vitals are
    source = extraction_json.get("vital_signs")\
        or extraction_json.get("extracted_vitals")\
        or extraction_json

    def _get_val(keys_list, target_type=float):
        """Search for a value across multiple possible keys and cast to *target_type*."""
        for key in keys_list:
            val = None
            # Nested dict (Rich format): try .value or .normalized_value_celsius
            if isinstance(source.get(key), dict):
                val = source[key].get("normalized_value_celsius") or source[key].get("value")
            # Direct value (Flat format)
            else:
                val = source.get(key)
            
            if val is not None:
                try:
                    return target_type(val)
                except (ValueError, TypeError):
                    continue  # Try the next key if conversion fails
        return None

    def _get_bp(component):
        """Extract a blood-pressure component (systolic / diastolic)."""
        # Try nested structure: blood_pressure -> systolic
        bp_obj = source.get("blood_pressure")
        if isinstance(bp_obj, dict):
            return bp_obj.get(component)
        # Try flat keys: sbp, systolic, etc.
        return _get_val([component, "sbp" if component == "systolic" else "dbp"])

    # 2. Mapping and normalisation
    normalized = {
        "heartrate": _get_val(["heartrate", "heart_rate", "hr"]),
        "resprate": _get_val(["resprate", "respiratory_rate", "rr"]),
        "temperature": _get_val(["temperature", "temp", "t"]),
        "o2sat": _get_val(["o2sat", "oxygen_saturation", "spo2", "sat"]),
        "sbp": _get_bp("systolic"),
        "dbp": _get_bp("diastolic"),
        "gcs": _get_val(["gcs", "glasgow"], int),
        "avpu": source.get("avpu") or source.get("acvpu")  # AVPU is typically a string
    }

    # 3. Supplemental oxygen logic (crucial for NEWS2)
    # If not explicitly detected, default to False (safety)
    supp_o2 = source.get("supplemental_oxygen")
    if isinstance(supp_o2, dict):
        # Complex extraction logic: "Room Air" -> False
        delivery = str(supp_o2.get("delivery_method", "")).lower()
        normalized["supplemental_oxygen"] = False if "room" in delivery\
            or "ambient" in delivery else True
    else:
        normalized["supplemental_oxygen"] = bool(supp_o2) if supp_o2 is not None\
            else False

    return normalized

def check_data_sufficiency(normalized_data: dict, score_type: str = "NEWS") -> List[str]:
    """
    Auxiliary function for the Auditor Agent.

    Returns a list of fields required for *score_type* that are missing
    from *normalized_data*, enabling early detection before calculation.
    """
    required = {
        "NEWS": ["resprate", "o2sat", "sbp", "heartrate", "temperature", "avpu"],
        "MEWS": ["resprate", "heartrate", "sbp", "temperature", "avpu"]
    }
    
    missing = []
    req_fields = required.get(score_type, [])
    
    for field in req_fields:
        if normalized_data.get(field) is None:
            # Logical fallback: if AVPU is missing but GCS exists, accept.
            if field == "avpu" and normalized_data.get("gcs") is not None:
                continue
            missing.append(field)
            
    return missing