from pydantic import BaseModel, Field
from typing import Optional

class InputSchema(BaseModel):
    """Schema limpo e estrito para entrada."""
    subject_id: int = Field(..., description="Unique patient identifier.")
    hadm_id: Optional[int] = Field(None, description="Hospital admission identifier.")
    raw_text: str = Field(..., description="Clinical note text.")