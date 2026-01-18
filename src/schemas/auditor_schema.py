from pydantic import BaseModel, Field
from typing import Literal

class AuditorEvaluation(BaseModel):
    """
    Represents the structured evaluation result produced by the Clinical Auditor agent,
    detailing compliance with official medical protocols.
    """
    
    protocol_reference: str = Field(..., description="Exact name of the protocol or paper utilized (e.g.: Sepsis-3 Guidelines).")

    clinical_risk_category: Literal['Low Risk / Stable', 'Medium Risk / Monitor', 'High Risk / Critical'] = Field(
        ...,
        description="Clinical Risk categorical classification based on the scores and findings."
    )
    
    evidence_quote: str = Field(..., description="Dicrect quote (ipsis litteris) of the recovered text that justify the judgement")
    clinical_suggestion: str = Field(..., description="Recommended action strictly based on the protocol.")

class AuditorOutput(BaseModel):
    """
    Container for the full output of the Auditor process, including search metadata
    and the final clinical evaluation decision.
    """
    search_query_used: str
    retrieved_context_summary: str
    evaluation: AuditorEvaluation