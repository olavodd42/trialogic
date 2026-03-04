"""Deterministic clinical-score calculator (NEWS2 and MEWS).

Provides hard-coded scoring logic to guarantee mathematical precision
where LLMs might hallucinate or miscalculate.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ScoreBreakdown(BaseModel):
    """
    Schema representing the detailed result of a clinical score calculation.
    """
    total_score: Optional[int] = None  # Can be None if data is insufficient
    breakdown: Dict[str, int]
    assumptions_used: List[str]
    is_estimate: bool = False
    status: str = "CALCULATED" # "CALCULATED", "INSUFFICIENT_DATA", "ERROR"

class ClinicalCalculator:
    """
    Deterministic tool for clinical score calculation.

    Implements hard-coded logic to ensure mathematical precision where
    LLMs might fail.
    """

    @staticmethod
    def calculate_news(vitals: dict) -> dict:
        """
        Calculates the NEWS2 (National Early Warning Score 2).

        Args:
            vitals (dict): Dictionary containing normalized vital signs.

        Returns:
            dict: The breakdown of the score and the total score.
        """
        logger.debug("Calculating NEWS2 score...")
        score = 0
        breakdown = {}
        missing_data = []

        # 1. Respiratory Rate
        rr = vitals.get('resprate')
        if rr is None:
            missing_data.append("resprate")
            s = 0
        elif rr <= 8: s = 3
        elif 9 <= rr <= 11: s = 1
        elif 12 <= rr <= 20: s = 0
        elif 21 <= rr <= 24: s = 2
        else: s = 3
        score += s
        breakdown['resprate'] = s

        # 2. SpO2 (Scale 1 - Assuming no hypercapnic failure for general screening)
        spo2 = vitals.get('o2sat')
        if spo2 is None:
            missing_data.append("o2sat")
            s = 0
        elif spo2 <= 91: s = 3
        elif 92 <= spo2 <= 93: s = 2
        elif 94 <= spo2 <= 95: s = 1
        else: s = 0
        score += s
        breakdown['o2sat'] = s

        # 3. Supplemental Oxygen
        supp_o2 = vitals.get('supplemental_oxygen', False)
        s = 2 if supp_o2 else 0
        score += s
        breakdown['supplemental_oxygen'] = s

        # 4. Temperature
        temp = vitals.get('temperature')
        if temp is None:
            missing_data.append("temperature")
            s = 0
        elif temp <= 35.0: s = 3
        elif 35.1 <= temp <= 36.0: s = 1
        elif 36.1 <= temp <= 38.0: s = 0
        elif 38.1 <= temp <= 39.0: s = 1
        else: s = 2 # > 39.1
        score += s
        breakdown['temperature'] = s

        # 5. Systolic BP
        sbp = vitals.get('sbp')
        if sbp is None:
            missing_data.append("sbp")
            s = 0
        elif sbp <= 90: s = 3
        elif 91 <= sbp <= 100: s = 2
        elif 101 <= sbp <= 110: s = 1
        elif 111 <= sbp <= 219: s = 0
        else: s = 3
        score += s
        breakdown['sbp'] = s

        # 6. Heart Rate
        hr = vitals.get('heartrate')
        if hr is None:
            missing_data.append("heartrate")
            s = 0
        elif hr <= 40: s = 3
        elif 41 <= hr <= 50: s = 1
        elif 51 <= hr <= 90: s = 0
        elif 91 <= hr <= 110: s = 1
        elif 111 <= hr <= 130: s = 2
        else: s = 3
        score += s
        breakdown['heartrate'] = s

        # 7. Consciousness (ACVPU)
        acvpu = vitals.get('acvpu', 'Alert').lower()
        avpu = vitals.get('avpu', 'Alert').lower()
        
        if acvpu == 'confusion' or avpu in ['voice', 'pain', 'unresponsive']:
            s = 3
        else:
            s = 0
            
        score += s
        breakdown['consciousness'] = s

        if not missing_data:
            logger.info("NEWS2 score calculated successfully: %s", score)
        else:
            logger.warning("Missing data: %s", missing_data)

        return {
            "score": score,
            "breakdown": breakdown,
            "risk": "High" if score >= 7 else "Medium" if score >= 5 else "Low",
            "missing_fields": missing_data
        }

    @staticmethod
    def calculate_mews(vitals: dict) -> dict:
        """Calculate the MEWS (Modified Early Warning Score).

        Args:
            vitals: Dictionary containing normalised vital signs.

        Returns:
            Dictionary with score breakdown, total, risk level, and missing fields.
        """
        logger.debug("Calculating MEWS score...")
        score = 0
        breakdown = {}
        missing_data = []

        # 1. RR
        rr = vitals.get('resprate')
        
        if rr is None:
            missing_data.append("resprate")
            s = 0
        elif rr < 9: s = 2
        elif 9 <= rr <= 14: s = 0
        elif 15 <= rr <= 20: s = 1
        elif 21 <= rr <= 29: s = 2
        else: s = 3
        score += s
        breakdown['resprate'] = s

        # 2. HR
        hr = vitals.get('heartrate')
        if hr is None:
            missing_data.append("heartrate")
            s = 0
        elif hr < 40: s = 2
        elif 41 <= hr <= 50: s = 1
        elif 51 <= hr <= 100: s = 0
        elif 101 <= hr <= 110: s = 1
        elif 111 <= hr <= 129: s = 2
        else: s = 3
        score += s
        breakdown['heartrate'] = s

        # 3. SBP
        sbp = vitals.get('sbp')
        if sbp is None:
            missing_data.append("sbp")
            s = 0
        elif sbp <= 70: s = 3
        elif 71 <= sbp <= 80: s = 2
        elif 81 <= sbp <= 100: s = 1
        elif 101 <= sbp <= 199: s = 0
        else: s = 2
        score += s
        breakdown['sbp'] = s

        # 4. Temp
        temp = vitals.get('temperature')
        if temp is None:
            missing_data.append("temperature")
            s = 0
        elif temp < 35: s = 2
        elif 35 <= temp <= 38.4: s = 0
        elif 38.5 <= temp < 39: s = 1 
        else: s = 2
        score += s
        breakdown['temperature'] = s

        # 5. AVPU (Neuro)
        
        avpu_raw = vitals.get('avpu', 'Alert').lower()
        acvpu_raw = vitals.get('acvpu', 'Alert').lower() 

        if avpu_raw == 'alert' and acvpu_raw == 'alert':
            s = 0
        elif avpu_raw == 'voice' or acvpu_raw == 'confusion': 
            s = 1 
        elif avpu_raw == 'pain':
            s = 2
        elif avpu_raw == 'unresponsive':
            s = 3
        else:
            s = 0 

        score += s
        breakdown['avpu'] = s
        logger.info("MEWS score calculated successfully: %s", score)

        return {
            "score": score,
            "breakdown": breakdown,
            "risk": "Critical" if score >= 5 else "Monitor",
            "missing_fields": missing_data
        }

def calculate_clinical_score(vitals: dict, score_name: str) -> str:
    """Entry-point used by the Mathematician agent to compute a clinical score.

    Args:
        vitals: Flat dictionary of normalised vital signs.
        score_name: ``"NEWS"`` or ``"MEWS"``.

    Returns:
        Human-readable string summarising the calculation result.
    """
    try:
        if score_name == "NEWS":
            result = ClinicalCalculator.calculate_news(vitals)
        elif score_name == "MEWS":
            result = ClinicalCalculator.calculate_mews(vitals)
        else:
            logger.error("Non-supported score: %s", score_name)
            return "ERROR: Non-supported score."

        if isinstance(result, dict):
            total_score = result.get("score")
            breakdown = result.get("breakdown", {})
            risk = result.get("risk", "Unknown")
            
            output = (
                f"STATUS: [CALCULATED]\n"
                f"SCORE TOTAL {score_name}: {total_score}\n"
                f"BREAKDOWN: {breakdown}\n"
                f"RISK: {risk}\n"
            )
            return output

        if hasattr(result, "status") and result.status == "INSUFFICIENT_DATA":
            logger.warning(
                f"STATUS: INSUFFICIENT DATA for {score_name}\n"
                f"MISSING FIELDS: {result.assumptions_used}\n"
                f"ACTION: Please search patient history or request vitals check."
            )
            return (
                f"STATUS: INSUFFICIENT DATA for {score_name}\n"
                f"MISSING FIELDS: {result.assumptions_used}\n"
                f"ACTION: Please search patient history or request vitals check."
            )
        
        status_tag = "[ESTIMATED]" if getattr(result, "is_estimate", False) else "[CONFIRMED]"
        total_score = getattr(result, "total_score", getattr(result, "score", "N/A"))
        breakdown = getattr(result, "breakdown", {})
        
        output = (
            f"STATUS: {status_tag}\n"
            f"SCORE TOTAL {score_name}: {total_score}\n"
            f"BREAKDOWN: {breakdown}\n"
        )
        assumptions = getattr(result, "assumptions_used", [])
        if assumptions:
            output += f"WARNING: {assumptions}\n"
            
        return output

    except Exception as e:
        logger.error("Error calculating %s: %s", score_name, e)
        return f"CRITICAL ERROR: {str(e)}"