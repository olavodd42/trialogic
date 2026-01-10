from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.schemas.scribe_output_schema import VitalsSchema
from src.schemas.mathematician_schema import ScoreCapability


# --- Schemas Auxiliares para Auditoria ---
class ScoreBreakdown(BaseModel):
    total_score: int
    breakdown: Dict[str, int]  # Ex: {"resprate": 3, "sbp": 0}
    assumptions_used: List[str]

# --- Refatoração da Lógica de Cálculo ---
class VitalSignCalculator:

    @staticmethod
    def _safe_get(value: Any, default: Any) -> Any:
        """Helper para evitar comparar None com int."""
        return value if value is not None else default

    @staticmethod
    def check_capability(vitals: 'VitalsSchema', score_name: str) -> 'ScoreCapability':
        # 1. Definição do que é obrigatório
        required_fields = {
            "resprate": vitals.resprate,
            "heartrate": vitals.heartrate,
            "sbp": vitals.sbp,
            "temperature": vitals.temperature,
        }

        # NEWS exige Saturação e O2 Suplementar
        if score_name == "NEWS":
            required_fields["o2sat"] = vitals.o2sat
            # O2 suplementar é booleano, se for None, assumimos False (mas avisamos)
            # Nota: 'supplemental_oxygen' deve existir no seu VitalsSchema

        # 2. Detecção de campos faltantes (que são estritamente numéricos)
        missing = [k for k, v in required_fields.items() if v is None]
        assumptions = []

        # 3. Regras de Negócio para Neurológico (MIMIC-IV Workaround)
        # Se for NEWS e faltar ACVPU/GCS
        if score_name == "NEWS" and not vitals.acvpu and not vitals.gcs:
            assumptions.append("Missing Neuro (ACVPU/GCS): Assumed 'Alert' (0 pts)")
        
        # Se for MEWS e faltar AVPU/GCS
        if score_name == "MEWS" and not vitals.avpu and not vitals.gcs:
            assumptions.append("Missing Neuro (AVPU/GCS): Assumed 'Alert' (0 pts)")
            
        # 4. Regra para O2 Suplementar (Se não constar, assume ar ambiente)
        if score_name == "NEWS" and getattr(vitals, 'supplemental_oxygen', None) is None:
             assumptions.append("Missing Supp O2: Assumed 'False' (Room Air)")

        return ScoreCapability(
            score_name=score_name,
            can_calculate=(len(missing) == 0),
            missing_fields=missing,
            assumptions_made=assumptions
        )

    @staticmethod
    def calculate_news(vitals: 'VitalsSchema', assumptions: List[str]) -> ScoreBreakdown:
        score = 0
        details = {}
        
        # Helper para evitar crash com None (usando valores normais fisiológicos como fallback neutro)
        # Nota: Só chegamos aqui se o check_capability permitiu, mas segurança nunca é demais.
        rr = VitalSignCalculator._safe_get(vitals.resprate, 18)
        hr = VitalSignCalculator._safe_get(vitals.heartrate, 75)
        sbp = VitalSignCalculator._safe_get(vitals.sbp, 120)
        temp = VitalSignCalculator._safe_get(vitals.temperature, 37.0)
        spo2 = VitalSignCalculator._safe_get(vitals.o2sat, 98)
        
        # Lógica O2 Suplementar
        supp_o2 = getattr(vitals, 'supplemental_oxygen', False)
        if supp_o2 is None: supp_o2 = False

        # --- Frequência Respiratória ---
        if rr <= 8: s = 3
        elif 9 <= rr <= 11: s = 1
        elif 12 <= rr <= 20: s = 0
        elif 21 <= rr <= 24: s = 2
        else: s = 3
        score += s; details['resprate'] = s

        # --- Saturação O2 (Escala 1 - Padrão) ---
        if spo2 <= 91: s = 3
        elif 92 <= spo2 <= 93: s = 2
        elif 94 <= spo2 <= 95: s = 1
        else: s = 0
        score += s; details['o2sat'] = s

        # --- Oxigênio Suplementar ---
        s = 2 if supp_o2 else 0
        score += s; details['supplemental_oxygen'] = s

        # --- Pressão Sistólica ---
        if sbp <= 90: s = 3
        elif 91 <= sbp <= 100: s = 2
        elif 101 <= sbp <= 110: s = 1
        elif 111 <= sbp <= 219: s = 0
        else: s = 3
        score += s; details['sbp'] = s

        # --- Frequência Cardíaca ---
        if hr <= 40: s = 3
        elif 41 <= hr <= 50: s = 1
        elif 51 <= hr <= 90: s = 0
        elif 91 <= hr <= 110: s = 1
        elif 111 <= hr <= 130: s = 2
        else: s = 3
        score += s; details['heartrate'] = s

        # --- Temperatura ---
        if temp <= 35.0: s = 3
        elif 35.1 <= temp <= 36.0: s = 1
        elif 36.1 <= temp <= 38.0: s = 0
        elif 38.1 <= temp <= 39.0: s = 1
        else: s = 2
        score += s; details['temperature'] = s

        # --- Consciência (ACVPU) ---
        # Lógica: Se tem assumption de neuro, é 0. Se não, verifica o dado real.
        neuro_score = 0
        if not any("Missing Neuro" in a for a in assumptions):
            # Se tivermos dados reais e NÃO for Alert, pontua 3
            status = str(vitals.acvpu).lower() if vitals.acvpu else "alert"
            if status != "alert":
                neuro_score = 3
        
        score += neuro_score; details['acvpu'] = neuro_score

        return ScoreBreakdown(total_score=score, breakdown=details, assumptions_used=assumptions)

    @staticmethod
    def calculate_mews(vitals: 'VitalsSchema', assumptions: List[str]) -> ScoreBreakdown:
        score = 0
        details = {}

        # 1. Sanitização de Inputs (Defensive Programming)
        # Valores padrão fisiológicos para não quebrar o cálculo se algo passou null
        rr = VitalSignCalculator._safe_get(vitals.resprate, 18)
        hr = VitalSignCalculator._safe_get(vitals.heartrate, 75)
        sbp = VitalSignCalculator._safe_get(vitals.sbp, 120)
        temp = VitalSignCalculator._safe_get(vitals.temperature, 37.0)

        # 2. Lógica AVPU (Neurológico)
        # Se assumimos que ele está Alerta (devido a dados faltantes), score é 0.
        if any("Missing Neuro" in a for a in assumptions):
            neuro_score = 0
        else:
            # Normaliza para lower case para evitar erro se vier "Voice" ou "voice"
            avpu_input = str(vitals.avpu).lower() if vitals.avpu else "alert"
            
            # Mapeamento Limpo (Clean Code)
            avpu_map = {
                "alert": 0,
                "a": 0,      # Caso venha abreviado
                "voice": 1,
                "v": 1,
                "pain": 2,
                "p": 2,
                "unresponsive": 3,
                "u": 3
            }
            # .get(key, default) -> Se vier algo bizarro, assume 3 (pior caso) ou lança erro. 
            # Clinicamente, assumir o pior em dados sujos é mais seguro, mas aqui assumiremos 3.
            neuro_score = avpu_map.get(avpu_input, 3) 

        score += neuro_score; details['avpu'] = neuro_score

        # 3. Frequência Respiratória (MEWS)
        if rr < 9: s = 2
        elif 9 <= rr <= 14: s = 0
        elif 15 <= rr <= 20: s = 1
        elif 21 <= rr <= 29: s = 2
        else: s = 3
        score += s; details['resprate'] = s

        # 4. Frequência Cardíaca (MEWS)
        if hr < 40: s = 2
        elif 40 <= hr <= 50: s = 1
        elif 51 <= hr <= 100: s = 0
        elif 101 <= hr <= 110: s = 1
        elif 111 <= hr <= 129: s = 2
        else: s = 3
        score += s; details['heartrate'] = s

        # 5. Pressão Sistólica (MEWS)
        if sbp < 70: s = 3
        elif 70 <= sbp <= 80: s = 2
        elif 81 <= sbp <= 100: s = 1
        elif 101 <= sbp <= 199: s = 0
        else: s = 2 # >= 200
        score += s; details['sbp'] = s

        # 6. Temperatura (MEWS)
        if temp < 35.0: s = 2
        elif 35.0 <= temp <= 38.4: s = 0 # Note: MEWS geralmente corta em 38.5
        else: s = 2 # >= 38.5
        score += s; details['temperature'] = s


        return ScoreBreakdown(total_score=score, breakdown=details, assumptions_used=assumptions)

# --- A Tool Definitiva ---

@tool
def calculate_clinical_score(vitals: 'VitalsSchema', score_name: str) -> str:
    """
    Calcula scores clínicos (NEWS ou MEWS) de forma determinística.
    Retorna uma string formatada com o score total e os detalhes.
    """
    # 1. Verifica Capacidade
    capability = VitalSignCalculator.check_capability(vitals, score_name)
    
    if not capability.can_calculate:
        return f"ERRO: Não é possível calcular {score_name}. Faltam dados: {capability.missing_fields}"

    # 2. Calcula
    try:
        if score_name == "NEWS":
            result = VitalSignCalculator.calculate_news(vitals, capability.assumptions_made)
        elif score_name == "MEWS":
            result = VitalSignCalculator.calculate_mews(vitals, capability.assumptions_made)
        else:
            return "ERRO: Score não suportado. Use 'NEWS' ou 'MEWS'."
            
        # 3. Retorna Formato Amigável para o LLM ler
        return (
            f"RESULTADO {score_name}: {result.total_score}\n"
            f"DETALHES: {result.breakdown}\n"
            f"NOTAS DE AUDITORIA: {result.assumptions_used}"
        )
    except Exception as e:
        return f"ERRO INTERNO DE CÁLCULO: {str(e)}"