from typing import List, Tuple, Dict, TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field, field_validator

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

    @field_validator("summary")
    def check_summary_length(cls, v):
        if v is not None and len(v.split()) > 300:
            raise ValueError("summary must be concise (under ~100 words)")
        return v

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

    @field_validator("temperature")
    def check_temperature(cls, v):
        if v is not None and not (25.0 <= v <= 45.0):
            raise ValueError("temperature must be between 25.0 and 45.0 Celsius")
        return v

    @field_validator("heartrate")
    def check_heartrate(cls, v):
        if v is not None and not (0 <= v <= 300):
            raise ValueError("heartrate must be between 0 and 300 BPM")
        return v
    
    @field_validator("resprate")
    def check_resprate(cls, v):
        if v is not None and not (0 < v <= 60):
            raise ValueError("resprate must be between 0 and 60 RPM")
        return v
    
    @field_validator("o2sat")
    def check_o2sat(cls, v):
        if v is not None and not (0 <= v <= 100):
            raise ValueError("o2sat must be between 0 and 100 %")
        return v
    
    @field_validator("sbp")
    def check_sbp(cls, v):
        if v is not None and not (10 <= v <= 400):
            raise ValueError("sbp must be between 10 and 400 mmHg")
        return v

    @field_validator("dbp")
    def check_dbp(cls, v):
        if v is not None and not (10 <= v <= 300):
            raise ValueError("dbp must be between 10 and 300 mmHg")
        return v
    
    @field_validator("pain")
    def check_pain(cls, v):
        if v is not None and not (0 <= v <= 10):
            raise ValueError("pain must be between 0 and 10")
        return v
    
    @field_validator("acuity")
    def check_acuity(cls, v):
        if v is not None and not (1 <= v <= 5):
            raise ValueError("acuity must be between 1 and 5")
        return v
    
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

    @field_validator("potassium")
    def check_potassium(cls, v):
        if v is not None and not (0.5 <= v <= 12.0):
            raise ValueError("potassium must be between 0.5 and 12.0 mEq/L")
        return v
    
    @field_validator("sodium")
    def check_sodium(cls, v):
        if v is not None and not (80.0 <= v <= 200.0):
            raise ValueError("sodium must be between 80.0 and 200.0 mEq/L")
        return v
    
    @field_validator("creatinine")
    def check_creatinine(cls, v):
        if v is not None and not (0.1 <= v <= 40.0):
            raise ValueError("creatinine must be between 0.1 and 40.0 mg/dL")
        return v
    
    @field_validator("wbc")
    def check_wbc(cls, v):
        if v is not None and not (0 <= v <= 500.0):
            raise ValueError("wbc must be between 0 and 500.0 K/uL")
        return v
    
    @field_validator("platelets")
    def check_platelets(cls, v):
        if v is not None and not (0 <= v <= 2000.0):
            raise ValueError("platelets must be between 0 and 2000.0 K/uL")
        return v
    
    @field_validator("inr")
    def check_inr(cls, v):
        if v is not None and not (0.5 <= v <= 30.0):
            raise ValueError("inr must be between 0.5 and 30.0")
        return v
    
    @field_validator("albumin")
    def check_albumin(cls, v):
        if v is not None and not (0.5 <= v <= 7.0):
            raise ValueError("albumin must be between 0.5 and 7.0 g/dL")
        return v
    
    @field_validator("bilirubin")
    def check_bilirubin(cls, v):
        if v is not None and not (0.1 <= v <= 80.0):
            raise ValueError("bilirubin must be between 0.1 and 80.0 mg/dL")
        return v
    
    @field_validator("ast")
    def check_ast(cls, v):
        if v is not None and not (0 <= v <= 20000):
            raise ValueError("ast must be between 0 and 20000 U/L")
        return v
    
    @field_validator("alt")
    def check_alt(cls, v):
        if v is not None and not (0 <= v <= 20000):
            raise ValueError("alt must be between 0 and 20000 U/L")
        return v
    

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
