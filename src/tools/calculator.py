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

class VitalSignCalculator:
    """
    Utility class for deterministic calculation of clinical scores (NEWS/MEWS).
    
    ACADEMIC NOTE:
    Implements 'Explicit Failure' pattern (Safe AI). 
    Critical missing values prevent score calculation instead of assuming normality (Imputation Bias).
    """

    @staticmethod
    def _safe_get(value: Optional[int | float]) -> Optional[int | float]:
        """Helper ensuring safe retrieval."""
        return value

    @staticmethod
    def calculate_news(vitals: VitalsSchema) -> ScoreBreakdown:
        score = 0
        breakdown = {}
        missing_fields = []

        # Helper lambda for concise logic
        # TECH LEAD FIX: Se valor é None, NÃO retorna 0. Retorna None para sinalizar falha.
        def get_points(val, logic_func, field_name):
            if val is None:
                missing_fields.append(field_name)
                return None 
            return logic_func(val)

        # 1. Respiration Rate (CRITICAL)
        # Note: Names aligned with VitalsSchema (resprate)
        rr_points = get_points(
            vitals.resprate, 
            lambda x: 3 if x <= 8 or x >= 25 else (2 if x >= 21 else (1 if x <= 11 else 0)), 
            "resprate"
        )
        if rr_points is not None:
            score += rr_points
            breakdown["resprate"] = rr_points

        # 2. O2 Saturation
        spo2_points = get_points(
            vitals.o2sat, 
            lambda x: 3 if x <= 91 else (2 if x <= 93 else (1 if x <= 95 else 0)), 
            "o2sat"
        )
        if spo2_points is not None:
            score += spo2_points
            breakdown["o2sat"] = spo2_points

        # 3. Supplemental Oxygen
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
        temp_points = get_points(
            vitals.temperature, 
            lambda x: 3 if x <= 35.0 else (1 if x <= 36.0 or x >= 38.1 else (2 if x >= 39.1 else 0)), 
            "temperature"
        )
        if temp_points is not None:
            score += temp_points
            breakdown["temperature"] = temp_points

        # 5. Systolic BP
        sbp_points = get_points(
            vitals.sbp, 
            lambda x: 3 if x <= 90 else (2 if x <= 100 else (1 if x <= 110 else 0)), 
            "sbp"
        )
        if sbp_points is not None:
            score += sbp_points
            breakdown["sbp"] = sbp_points


        # 6. Heart Rate
        hr_points = get_points(
            vitals.heartrate, 
            lambda x: 3 if x <= 40 or x >= 131 else (2 if x >= 111 else (1 if x <= 50 or x >= 91 else 0)), 
            "heartrate"
        )
        if hr_points is not None:
            score += hr_points
            breakdown["heartrate"] = hr_points

        # 7. AVPU/Consciousness
        points = 0
        avpu = str(vitals.avpu).lower() if vitals.avpu else ""
        
        if not avpu:
            # Se não tem info de consciência, é crítico.
            missing_fields.append("consciousness")
        elif "alert" in avpu:
            points = 0
        elif any(x in avpu for x in ["verbal", "pain", "unresponsive", "voice"]):
            points = 3
        else:
            points = 0 # Default safe fallback if text is weird, but could be strict.
        
        if "consciousness" not in missing_fields:
            score += points
            breakdown["consciousness"] = points

        # --- SAFETY CHECK ---
        # Se campos críticos faltam, INVALIDAMOS o score.
        if missing_fields:
            return ScoreBreakdown(
                total_score=None,
                breakdown=breakdown,
                assumptions_used=[f"MISSING CRITICAL DATA: {f}" for f in missing_fields],
                is_estimate=True,
                status="INSUFFICIENT_DATA"
            )
        
        return ScoreBreakdown(
            total_score=score,
            breakdown=breakdown,
            assumptions_used=[],
            is_estimate=False,
            status="CALCULATED"
        )
    
    @staticmethod
    def calculate_mews(vitals: VitalsSchema) -> ScoreBreakdown:
        score = 0
        breakdown = {}
        missing_fields = []

        def get_points(val, logic_func, field_name):
            if val is None:
                missing_fields.append(field_name)
                return None
            return logic_func(val)

        # MEWS Logic (Simplified Standard)
        rr_points = get_points(
            vitals.resprate, 
            lambda x: 2 if x <= 8 or x >= 21 else (3 if x >= 30 else 0), 
            "resprate"
        )
        if rr_points is not None: score += rr_points; breakdown["resprate"] = rr_points

        hr_points = get_points(
            vitals.heartrate, 
            lambda x: 2 if x <= 40 or x >= 111 else (3 if x >= 130 else (1 if x <= 50 or x >= 101 else 0)), 
            "heartrate"
        )
        if hr_points is not None: score += hr_points; breakdown["heartrate"] = hr_points

        sbp_points = get_points(
            vitals.sbp, 
            lambda x: 3 if x <= 70 else (2 if x <= 80 else (1 if x <= 100 else 0)), 
            "sbp"
        )
        if sbp_points is not None: score += sbp_points; breakdown["sbp"] = sbp_points

        temp_points = get_points(
            vitals.temperature, 
            lambda x: 2 if x <= 35.0 or x >= 38.5 else 0, 
            "temperature"
        )
        if temp_points is not None: score += temp_points; breakdown["temperature"] = temp_points

        # AVPU
        avpu = str(vitals.avpu).lower() if vitals.avpu else ""
        points = 0
        if not avpu:
            missing_fields.append("avpu")
        elif "alert" in avpu: points = 0
        elif "verbal" in avpu: points = 1
        elif "pain" in avpu: points = 2
        elif "unresponsive" in avpu: points = 3
        
        if "avpu" not in missing_fields:
            score += points
            breakdown["avpu"] = points

        if missing_fields:
            return ScoreBreakdown(
                total_score=None,
                breakdown=breakdown,
                assumptions_used=[f"MISSING: {f}" for f in missing_fields],
                is_estimate=True,
                status="INSUFFICIENT_DATA"
            )

        return ScoreBreakdown(
            total_score=score,
            breakdown=breakdown,
            assumptions_used=[],
            is_estimate=False,
            status="CALCULATED"
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
        
        # Formata a saída baseada no Status
        if result.status == "INSUFFICIENT_DATA":
            return (
                f"STATUS: INSUFFICIENT DATA for {score_name}\n"
                f"MISSING FIELDS: {result.assumptions_used}\n"
                f"ACTION: Please search patient history or request vitals check."
            )
        
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