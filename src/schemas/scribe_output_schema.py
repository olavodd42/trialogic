from typing import List, Tuple, Dict, TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

# --- Enums para Restrição de Output ---
# Isso impede que o modelo invente status como "Semi-Conscious"
TypeAdmissionType = Literal['EMERGENCY', 'URGENT', 'ELECTIVE']
TypeService = Literal['MEDICINE', 'SURGERY', 'ORTHOPEDICS', 'NEUROLOGY', 'PSYCHIATRY', 'OTHER']
TypeAVPU = Literal['Alert', 'Voice', 'Pain', 'Unresponsive']
TypeChange = Literal['NEW_START', 'DOSE_INCREASE', 'DOSE_DECREASE', 'STOPPED', 'UNCHANGED']

# --- Schemas de Output ---

class MetadataSchema(BaseModel):
    """Schema for administrative metadata."""
    admission_type: Optional[TypeAdmissionType] = Field(None, description="The urgency level of admission.")
    service: Optional[TypeService] = Field(None, description="Primary medical service.")
    discharge_disposition: Optional[str] = Field(None, description="Final disposition (e.g., 'Home', 'SNF', 'Expired').")

class SemanticSchema(BaseModel):
    """High-level semantic understanding of the visit."""
    summary: Optional[str] = Field(None, description="Concise 2-3 sentence clinical summary of the patient's course.")
    key_conditions: List[str] = Field(default_factory=list, description="List of active diagnoses and comorbidities managed.")
    chief_complaint: Optional[str] = Field(None, description="Primary symptom prompting the ED visit.")

class VitalsSchema(BaseModel):
    """Vital signs at triage/admission. Use None if not explicitly stated."""
    temperature: Optional[float] = Field(None, description="Temperature in Celsius. Convert from F if needed.")
    heartrate: Optional[int] = Field(None, description="Heart rate (BPM).")
    resprate: Optional[int] = Field(None, description="Respiratory rate (BPM).")
    o2sat: Optional[int] = Field(None, description="O2 Saturation (%) on admission.")
    sbp: Optional[int] = Field(None, description="Systolic BP (mmHg).")
    dbp: Optional[int] = Field(None, description="Diastolic BP (mmHg).")
    pain: Optional[int] = Field(None, description="Pain score (0-10).")
    acuity: Optional[int] = Field(None, description="ESI Acuity Level (1-5) if stated or clearly inferable.")
    supplemental_oxygen: Optional[bool] = Field(None, description="True if on O2, False if Room Air.")
    avpu: Optional[TypeAVPU] = Field(None, description="Level of consciousness.")

class LabsSchema(BaseModel):
    """Key laboratory values. Only extract pertinent abnormalities or admission baselines."""
    potassium: Optional[float] = Field(None, description="K+ (mEq/L)")
    sodium: Optional[float] = Field(None, description="Na+ (mEq/L)")
    creatinine: Optional[float] = Field(None, description="Cr (mg/dL)")
    wbc: Optional[float] = Field(None, description="WBC count (K/uL)")
    platelets: Optional[float] = Field(None, description="Plt count (K/uL)")
    inr: Optional[float] = Field(None, description="INR")
    albumin: Optional[float] = Field(None, description="Albumin (g/dL)")
    bilirubin: Optional[float] = Field(None, description="Total Bilirubin (mg/dL)")
    ast: Optional[float] = Field(None, description="AST (U/L)")
    alt: Optional[float] = Field(None, description="ALT (U/L)")

class FindingsSchema(BaseModel):
    """Physical exam findings focuses."""
    abdomen: Optional[str] = Field(None, description="Specific abdominal findings (e.g., 'distended', 'soft').")
    extremities: Optional[str] = Field(None, description="Edema, pulses, or perfusion findings.")
    mental_status: Optional[str] = Field(None, description="Detailed mental status exam notes.")

class ClinicalSchema(BaseModel):
    """Aggregated clinical metrics."""
    vitals: VitalsSchema = Field(..., description="Admission vital signs.")
    labs: LabsSchema = Field(..., description="Pertinent lab results.")
    findings: FindingsSchema = Field(..., description="Physical exam highlights.")

class DrugSchema(BaseModel):
    drug: str = Field(..., description="Generic name of the medication.")
    dose: Optional[str] = Field(None, description="Dose with units (e.g., '40 mg').")
    route: Optional[str] = Field(None, description="Route (PO, IV, IM).")
    frequency: Optional[str] = Field(None, description="Frequency (e.g., 'DAILY', 'BID').")

class DeltaSchema(BaseModel):
    """Analysis of medication changes."""
    drug: str = Field(..., description="Name of the drug changed.")
    change_type: TypeChange = Field(..., description="Nature of the change.")
    from_dose: Optional[str] = Field(None, description="Dose prior to admission/change.")
    to_dose: Optional[str] = Field(None, description="Dose at discharge.")
    reason_inferred: Optional[str] = Field(None, description="Inferred clinical reasoning for the change.")

class TreatmentsSchema(BaseModel):
    """Medication reconciliation."""
    admission_drugs: List[DrugSchema] = Field(default_factory=list, description="Home meds or meds on arrival.")
    discharge_drugs: List[DrugSchema] = Field(default_factory=list, description="Prescriptions at discharge.")
    delta_analysis: List[DeltaSchema] = Field(default_factory=list, description="Reconciliation of changes.")

class ScribeOutputSchema(BaseModel):
    """Master Schema for The Scribe Output."""
    metadata: MetadataSchema
    semantics: SemanticSchema
    clinical: ClinicalSchema
    treatments: TreatmentsSchema