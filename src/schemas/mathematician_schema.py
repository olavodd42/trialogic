from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ScoreCapability(BaseModel):
    """
    Represents the capability to calculate a specific clinical score.
    Critical for 'Glass Box' auditing - explains WHY a score was or wasn't calculated.
    """
    score_name: str = Field(..., description="Name of the clinical score (e.g., 'MEWS', 'NEWS2').")
    can_calculate: bool = Field(..., description="Indicates if sufficient data was present in the Python calculation.")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing fields if calculation failed.")
    assumptions_made: List[str] = Field(default_factory=list, description="List of assumptions made (e.g., 'Assumed Room Air for O2Sat').")

class RiskScoreAnalysis(BaseModel):
    """
    Detailed analysis of a successfully calculated score.
    """
    score_name: str = Field(..., description="Name of the score.")
    total_score: float = Field(..., description="The calculated numerical value.")
    risk_level: str = Field(..., description="Textual risk level (Low, Medium, High/Critical).")
    clinical_implication: str = Field(..., description="Brief clinical implication of this score.")

class MathematicianSchema(BaseModel):
    """
    Master Output Schema for the Mathematician Agent.
    Combines technical auditing (ScoreCapability) with clinical reasoning (RiskScoreAnalysis).
    """
    # Auditoria Técnica: O que foi possível calcular?
    capabilities: List[ScoreCapability] = Field(..., description="Audit report of calculation capabilities for each requested score.")
    
    # Análise Clínica: Dos que foram calculados, qual o risco?
    analyzed_scores: List[RiskScoreAnalysis] = Field(default_factory=list, description="List of scores that were successfully calculated and analyzed.")
    
    # Síntese Final
    overall_risk_assessment: str = Field(..., description="Synthesized assessment of the patient's stability based on all scores.")
    
    # Campo para injetar os dados brutos (Opcional, preenchido via código, não pela LLM)
    calculated_raw: Optional[Dict[str, Any]] = Field(None, description="Raw Python calculation results for debugging.")