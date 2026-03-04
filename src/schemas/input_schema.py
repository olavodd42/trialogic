"""Pydantic schema for raw clinical-note input data."""

from pydantic import BaseModel, Field
from typing import Optional

class InputSchema(BaseModel):
    """
    Schema representing the input data structure for clinical notes processing.
    Attributes:
        subject_id (int): Unique patient identifier.
        hadm_id (Optional[int]): Hospital admission identifier. Defaults to None.
        raw_text (str): The raw text content of the clinical note.
    """
    
    subject_id: int = Field(..., description="Unique patient identifier.")
    hadm_id: Optional[int] = Field(None, description="Hospital admission identifier.")
    raw_text: str = Field(..., description="Clinical note text.")