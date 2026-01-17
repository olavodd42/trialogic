import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from src.schemas.scribe_schema import VitalsSchema

logger = logging.getLogger(__name__)

class ScoreBreakdown(BaseModel):
    """
    Schema representing the detailed result of a clinical score calculation.
    """
    total_score: int
    breakdown: Dict[str, int]
    assumptions_used: List[str]
    is_estimate: bool = False

class VitalSignCalculator:
    """
    Utility class for deterministic calculation of clinical scores (NEWS/MEWS).
    Implements 'Imputation of Normalcy' pattern: missing values default to 0 (Normal).
    """

    @staticmethod
    def _safe_get(value: Optional[int | float]) -> Optional[int | float]:
        """Helper ensuring safe retrieval."""
        return value

    @staticmethod
    def calculate_news(vitals: 'VitalsSchema') -> ScoreBreakdown:
        score = 0
        breakdown = {}
        missing_fields = []

        # Helper lambda for concise logic
        # Se valor é None, pontos = 0, adiciona a missing
        def get_points(val, logic_func, field_name):
            if val is None:
                missing_fields.append(field_name)
                return 0
            return logic_func(val)

        # 1. Respiration Rate
        rr_points = get_points(vitals.resprate, lambda x: 3 if x <= 8 or x >= 25 else (2 if x >= 21 else (1 if x <= 11 else 0)), "resprate")
        score += rr_points
        breakdown["resprate"] = rr_points

        # 2. O2 Saturation
        spo2_points = get_points(vitals.o2sat, lambda x: 3 if x <= 91 else (2 if x <= 93 else (1 if x <= 95 else 0)), "o2sat")
        score += spo2_points
        breakdown["o2sat"] = spo2_points

        # 3. Supplemental Oxygen (Boolean/String handling)
        supp_o2 = vitals.supplemental_oxygen
        if supp_o2 is None:
            supp_o2_points = 0
        elif str(supp_o2).lower() in ['true', 'yes', 'sim', '1']:
            supp_o2_points = 2
        else:
            supp_o2_points = 0
        score += supp_o2_points
        breakdown["supplemental_oxygen"] = supp_o2_points

        # 4. Temperature
        temp_points = get_points(vitals.temperature, lambda x: 3 if x <= 35.0 else (1 if x <= 36.0 or x >= 38.1 else (2 if x >= 39.1 else 0)), "temperature")
        score += temp_points
        breakdown["temperature"] = temp_points

        # 5. Systolic BP
        sbp_points = get_points(vitals.sbp, lambda x: 3 if x <= 90 else (2 if x <= 100 else (1 if x <= 110 else 0)), "sbp")
        score += sbp_points
        breakdown["sbp"] = sbp_points

        # 6. Heart Rate
        hr_points = get_points(vitals.heartrate, lambda x: 3 if x <= 40 or x >= 131 else (2 if x >= 111 else (1 if x <= 50 or x >= 91 else 0)), "heartrate")
        score += hr_points
        breakdown["heartrate"] = hr_points

        # 7. AVPU/Consciousness
        # Simplification: Anything not 'Alert' or GCS < 15 is 3 points
        points = 0
        avpu = str(vitals.avpu).lower() if vitals.avpu else ""
        gcs = vitals.gcs
        if "alert" in avpu or (gcs == 15):
            points = 0
        elif avpu in ["verbal", "pain", "unresponsive"] or (gcs is not None and gcs < 15):
            points = 3
        elif not avpu and gcs is None:
            missing_fields.append("consciousness")
        
        score += points
        breakdown["consciousness"] = points

        assumptions = [f"Missing {f} -> Assumed Normal (0)" for f in missing_fields]
        
        return ScoreBreakdown(
            total_score=score,
            breakdown=breakdown,
            assumptions_used=assumptions,
            is_estimate=len(missing_fields) > 0
        )

    @staticmethod
    def calculate_mews(vitals: 'VitalsSchema') -> ScoreBreakdown:
        score = 0
        breakdown = {}
        missing_fields = []

        def get_points(val, logic_func, field_name):
            if val is None:
                missing_fields.append(field_name)
                return 0
            return logic_func(val)

        # MEWS Logic (Simplified Standard)
        rr_points = get_points(vitals.resprate, lambda x: 2 if x <= 8 or x >= 21 else (3 if x >= 30 else 0), "resprate")
        score += rr_points
        breakdown["resprate"] = rr_points

        hr_points = get_points(vitals.heartrate, lambda x: 2 if x <= 40 or x >= 111 else (3 if x >= 130 else (1 if x <= 50 or x >= 101 else 0)), "heartrate")
        score += hr_points
        breakdown["heartrate"] = hr_points

        sbp_points = get_points(vitals.sbp, lambda x: 3 if x <= 70 else (2 if x <= 80 else (1 if x <= 100 else 0)), "sbp")
        score += sbp_points
        breakdown["sbp"] = sbp_points

        temp_points = get_points(vitals.temperature, lambda x: 2 if x <= 35.0 or x >= 38.5 else 0, "temperature")
        score += temp_points
        breakdown["temperature"] = temp_points

        avpu = str(vitals.avpu).lower() if vitals.avpu else ""
        points = 0
        if "alert" in avpu: points = 0
        elif "verbal" in avpu: points = 1
        elif "pain" in avpu: points = 2
        elif "unresponsive" in avpu: points = 3
        else: missing_fields.append("avpu")
        
        score += points
        breakdown["avpu"] = points

        assumptions = [f"Missing {f} -> Assumed Normal (0)" for f in missing_fields]

        return ScoreBreakdown(
            total_score=score,
            breakdown=breakdown,
            assumptions_used=assumptions,
            is_estimate=len(missing_fields) > 0
        )

def calculate_clinical_score(vitals: VitalsSchema, score_name: str) -> str:
    """Entry point for the agent."""
    try:
        if score_name == "NEWS":
            result = VitalSignCalculator.calculate_news(vitals)
        elif score_name == "MEWS":
            result = VitalSignCalculator.calculate_mews(vitals)
        else:
            return "ERRO: Score não suportado."
        
        status_tag = "[ESTIMATED]" if result.is_estimate else "[CONFIRMED]"
        
        output = (
            f"STATUS: {status_tag}\n"
            f"SCORE TOTAL {score_name}: {result.total_score}\n"
            f"BREAKDOWN: {result.breakdown}\n"
        )
        if result.assumptions_used:
            output += f"WARNING: {result.assumptions_used}\n"
            
        return output

    except Exception as e:
        logger.error(f"Error calculating {score_name}: {e}")
        return f"CRITICAL ERROR: {str(e)}"