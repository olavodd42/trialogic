from typing import Optional, List, Dict, Any
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
    missing_fields: List[str] = Field(default_factory=list, description="List of missing fields required for calculation.")
    assumptions_made: List[str] = Field(default_factory=list, description="List of assumptions made during calculation, e.g., assuming 'Alert' for consciousness level.")
