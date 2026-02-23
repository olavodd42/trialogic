from typing import List, Optional, Mapping
from pydantic import BaseModel, Field

class ScoreCapability(BaseModel):
    """
    Represents the capability to calculate a specific clinical score.
    Attributes:
        score_name (str): Name of the clinical score (e.g., 'MEWS', 'NEWS2').
        can_calculate (bool): Indicates if sufficient data is present to calculate the score.
        missing_fields (List[str]): List of missing fields required for calculation.
        assumptions_made (List[str]): List of assumptions made during calculation, e.g., assuming 'Alert' for consciousness level.
    """
    score_name: str = Field(..., description="Name of the clinical score (e.g., 'MEWS', 'NEWS2').")
    can_calculate: bool = Field(..., description="Indicates if sufficient data is present to calculate the score.")
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
    # Technical Audit: What was possible to calculate?
    capabilities: List[ScoreCapability] = Field(..., description="Audit report of calculation capabilities for each requested score.")
    
    # Clinical Analysis: From those calculated, what is the risk?
    analyzed_scores: List[RiskScoreAnalysis] = Field(default_factory=list, description="List of scores that were successfully calculated and analyzed.")
    
    # Final Synthesis
    overall_risk_assessment: str = Field(..., description="Synthesized assessment of the patient's stability based on all scores.")
    
    # Field to inject raw data (Optional, filled via code, not by LLM)
    calculated_raw: Optional[Mapping[str, float]] = Field(None, description="Raw Python calculation results for debugging.")

    class Config:
        extra = "ignore"