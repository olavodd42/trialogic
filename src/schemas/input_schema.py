from pydantic import BaseModel, Field
from typing import Optional

class InputSchema(BaseModel):
    """Schema for input data to Scribe ED model."""
    subject_id: int = Field(description="unique identifier which specifies an individual patient.")
    hadm_id: Optional[int] = Field(description="If the patient was admitted to the hospital after their ED stay, \
        hadm_id will contain the hospital identifier (ranges from 2000000 - 2999999).")
    raw_text: str = Field(description="The clinical notes written by the physician at the time of discharge\
                                from the emergency department.")  
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)