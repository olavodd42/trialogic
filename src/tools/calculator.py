import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from src.schemas.scribe_schema import VitalsSchema

logger = logging.getLogger(__name__)

# --- TECH LEAD NOTE ---
# Alteramos o ScoreBreakdown para permitir total_score=None.
# Isso reflete o estado "INSUFFICIENT_DATA" academicamente correto.

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
    Ferramenta determinística para cálculo de scores clínicos.
    Implementa lógica 'Hard-Coded' para garantir precisão matemática que LLMs falham.
    """

    @staticmethod
    def calculate_news(vitals: dict) -> dict:
        logger.debug("Calculating NEWS2 score...")
        score = 0
        breakdown = {}

        # 1. Respiratory Rate
        rr = vitals.get('resprate')
        if rr is not None:
            if rr <= 8: s = 3
            elif 9 <= rr <= 11: s = 1
            elif 12 <= rr <= 20: s = 0
            elif 21 <= rr <= 24: s = 2
            else: s = 3
            score += s
            breakdown['resprate'] = s

        # 2. SpO2 (Scale 1 - Assuming no hypercapnic failure for general screening)
        spo2 = vitals.get('o2sat')
        if spo2 is not None:
            if spo2 <= 91: s = 3
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
        if temp is not None:
            if temp <= 35.0: s = 3
            elif 35.1 <= temp <= 36.0: s = 1
            elif 36.1 <= temp <= 38.0: s = 0
            elif 38.1 <= temp <= 39.0: s = 1
            else: s = 2 # > 39.1
            score += s
            breakdown['temperature'] = s

        # 5. Systolic BP
        sbp = vitals.get('sbp')
        if sbp is not None:
            if sbp <= 90: s = 3
            elif 91 <= sbp <= 100: s = 2
            elif 101 <= sbp <= 110: s = 1
            elif 111 <= sbp <= 219: s = 0
            else: s = 3
            score += s
            breakdown['sbp'] = s

        # 6. Heart Rate
        hr = vitals.get('heartrate')
        if hr is not None:
            if hr <= 40: s = 3
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

        logger.info(f"NEWS2 score calculated succesfully: {score}")

        return {
            "score": score,
            "breakdown": breakdown,
            "risk": "High" if score >= 7 else "Medium" if score >= 5 else "Low"
        }

    @staticmethod
    def calculate_mews(vitals: dict) -> dict:
        logger.debug("Calculating MEWS score...")
        score = 0
        breakdown = {}

        # 1. RR
        rr = vitals.get('resprate')
        if rr is not None:
            if rr < 9: s = 2
            elif 9 <= rr <= 14: s = 0
            elif 15 <= rr <= 20: s = 1
            elif 21 <= rr <= 29: s = 2
            else: s = 3
            score += s
            breakdown['resprate'] = s

        # 2. HR
        hr = vitals.get('heartrate')
        if hr is not None:
            if hr < 40: s = 2
            elif 41 <= hr <= 50: s = 1
            elif 51 <= hr <= 100: s = 0
            elif 101 <= hr <= 110: s = 1
            elif 111 <= hr <= 129: s = 2
            else: s = 3
            score += s
            breakdown['heartrate'] = s

        # 3. SBP
        sbp = vitals.get('sbp')
        if sbp is not None:
            if sbp <= 70: s = 3
            elif 71 <= sbp <= 80: s = 2
            elif 81 <= sbp <= 100: s = 1
            elif 101 <= sbp <= 199: s = 0
            else: s = 2
            score += s
            breakdown['sbp'] = s

        # 4. Temp
        temp = vitals.get('temperature')
        if temp is not None:
            if temp < 35: s = 2
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
        logger.info(f"MEWS score calculated succesfully: {score}")

        return {
            "score": score,
            "breakdown": breakdown,
            "risk": "Critical" if score >= 5 else "Monitor"
        }

def calculate_clinical_score(vitals: dict, score_name: str) -> str:

    """Entry point for the agent."""
    try:
        if score_name == "NEWS":
            result = ClinicalCalculator.calculate_news(vitals)
        elif score_name == "MEWS":
            result = ClinicalCalculator.calculate_mews(vitals)
        else:
            logger.error("Non-supported score.")
            return "ERRO: Non-supported score."

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
        logger.error(f"Error calculating {score_name}: {e}")
        return f"CRITICAL ERROR: {str(e)}"