from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class AuditorOutput(BaseModel):
    """
    Estrutura final de decisão clínica auditável.
    Projetado para garantir rastreabilidade entre dados, regras e decisão.
    """
    
    protocol_reference: str = Field(
        ..., 
        description="Exact name of the protocol used as base (ex: NEWS2, Sepsis-3, Manchester)."
    )
    
    clinical_risk_category: Literal['Low Risk', 'Medium Risk / Monitor', 'High Risk / Emergency', 'Critical / Resuscitation'] = Field(
        ...,
        description="Standardized risk category based on computed score and clinical context."
    )
    
    calculated_score_audit: str = Field(
        ...,
        description="Text confirmation of received score (e.g.: 'NEWS2 Score of 5 verified')."
    )
    
    evidence_quote: str = Field(
        ...,
        description="Direct and exact citation of the text retrieved by RAG that justifies the suggested conduct."
    )
    
    clinical_suggestion: str = Field(
        ...,
        description="Recommendation of immediate action for the medical/nursing team, concise and direct."
    )
    
    reasoning_trace: str = Field(
        ...,
        description="Chain of Thought resumed: Data + Rule = Conclusion."
    )

    missing_info_warning: Optional[str] = Field(
        None,
        description="Warning about crucial data that were absent on original inputl (e.g.: 'Lactate not reported.')."
    )