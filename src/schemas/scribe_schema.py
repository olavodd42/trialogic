from typing import List, Optional, Literal, Union, Any
from pydantic import BaseModel, Field

# --- Enums e Tipos ---
TypeAdmissionType = Literal['EMERGENCY', 'URGENT', 'ELECTIVE']
TypeService = Literal['MEDICINE', 'SURGERY', 'ORTHOPEDICS', 'NEUROLOGY', 'PSYCHIATRY', 'OTHER']
TypeAVPU = Literal['Alert', 'Voice', 'Pain', 'Unresponsive']
TypeACVPU = Literal['Alert', 'Confusion', 'Voice', 'Pain', 'Unresponsive']
TypeChange = Literal['NEW_START', 'DOSE_INCREASE', 'DOSE_DECREASE', 'STOPPED', 'UNCHANGED']

class RawVitalsLLM(BaseModel):
    resprate: Optional[int]
    heartrate: Optional[int]
    temperature: Optional[float] 
    o2sat: Optional[int]
    supplemental_oxygen: Optional[bool]
    sbp: Optional[int]
    dbp: Optional[int]
    avpu: Optional[TypeAVPU]
    acvpu: Optional[TypeACVPU]
    acuity: Optional[int]

class RawScribeLLM(BaseModel):
    chief_complaint: Optional[str]
    vitals: RawVitalsLLM


class MetadataSchema(BaseModel):
    admission_type: Optional[TypeAdmissionType] = Field(None, description="The urgency level of admission.")
    service: Optional[TypeService] = Field(None, description="Primary medical service.")
    discharge_disposition: Optional[str] = Field(None, description="Patient destination after discharge.")

class VitalsSchema(BaseModel):
    """
    Schema de Vitais Blindado e Completo.
    """
    
    heartrate: Optional[int] = Field(None, description="Heart rate (HR/BPM).")
    resprate: Optional[int] = Field(None, description="Respiratory Rate (RR) in rpm.")
    temperature: Optional[float] = Field(
        None,
        description="Temperature (Temp). Extract numeric value as float, e.g. 39.2."
    )

    o2sat: Optional[int] = Field(None, description="Oxygen Saturation (SpO2) %.")
    sbp: Optional[int] = Field(None, description="Systolic BP (top number).")
    dbp: Optional[int] = Field(None, description="Diastolic BP (bottom number).")
    
    acuity: Optional[int] = Field(None, description="ESI Acuity Scale (1-5).")
    
    supplemental_oxygen: Optional[bool] = Field(None, description="True if patient is on O2.")
    avpu: Optional[TypeAVPU] = Field(None, description="AVPU scale status.")
    acvpu: Optional[TypeACVPU] = Field(None, description="ACVPU (NEWS2) status.")


class ClinicalSchema(BaseModel):
    chief_complaint: Optional[str] = Field(None, description="Primary patient complaint.")
    vitals: VitalsSchema = Field(..., description="Vital signs measurements.")