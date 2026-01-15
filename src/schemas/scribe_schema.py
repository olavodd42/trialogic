from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

# --- Enums para Restrição de Output ---
# Isso impede que o modelo invente status como "Semi-Conscious"
TypeAdmissionType = Literal['EMERGENCY', 'URGENT', 'ELECTIVE']
TypeService = Literal['MEDICINE', 'SURGERY', 'ORTHOPEDICS', 'NEUROLOGY', 'PSYCHIATRY', 'OTHER']
TypeAVPU = Literal['Alert', 'Voice', 'Pain', 'Unresponsive']
TypeACVPU = Literal['Alert', 'Confusion', 'Voice', 'Pain', 'Unresponsive']
TypeChange = Literal['NEW_START', 'DOSE_INCREASE', 'DOSE_DECREASE', 'STOPPED', 'UNCHANGED']

# --- Schemas de Output ---

class MetadataSchema(BaseModel):
    """
    Schema for administrative metadata captured during the triage or admission process.
    
    Attributes:
        admission_type (TypeAdmissionType, optional): The urgency/category of the admission (EMERGENCY, URGENT, ELECTIVE).
        service (TypeService, optional): The primary medical specialty service responsible for the patient.
        discharge_disposition (str, optional): The destination or status of the patient upon discharge (e.g., 'Home', 'Expired').
    """
    admission_type: Optional[TypeAdmissionType] = Field(None, description="The urgency level of admission.")
    service: Optional[TypeService] = Field(None, description="Primary medical service.")
    discharge_disposition: Optional[str] = Field(None, description="Final disposition (e.g., 'Home', 'SNF', 'Expired').")

class SemanticSchema(BaseModel):
    """
    Schema for high-level semantic understanding and summarization of the patient's visit.
    
    Attributes:
        summary (str, optional): A concise clinical narrative summarizing the patient's presentation and course.
        key_conditions (List[str]): Extracted diagnoses, comorbidities, or acute problems managed during the visit.
        chief_complaint (str, optional): The primary reason for the visit or the main symptom reported by the patient.
    """
    summary: Optional[str] = Field(None, description="Concise 2-3 sentence clinical summary of the patient's course.")
    key_conditions: List[str] = Field(default_factory=list, description="List of active diagnoses and comorbidities managed.")
    chief_complaint: Optional[str] = Field(None, description="Primary symptom prompting the ED visit.")

class VitalsSchema(BaseModel):
    """
    Schema for structured vital signs data extracted from triage or admission notes.
    Fields should be None if the value is not explicitly stated in the source text.
    
    Attributes:
        temperature (float, optional): Body temperature in Celsius.
        heartrate (int, optional): Heart rate in beats per minute (BPM).
        resprate (int, optional): Respiratory rate in breaths per minute (RPM).
        o2sat (int, optional): Oxygen saturation percentage (SpO2).
        sbp (int, optional): Systolic blood pressure in mmHg.
        dbp (int, optional): Diastolic blood pressure in mmHg.
        pain (int, optional): Pain score on a scale of 0-10.
        acuity (int, optional): Triage acuity level (e.g., ESI 1-5).
        supplemental_oxygen (bool, optional): Boolean flag indicating if the patient is on supplemental oxygen.
        gcs (int, optional): Glasgow Coma Scale score (3-15).
        avpu (TypeAVPU, optional): Consciousness level based on the AVPU scale (MEWS standard).
        acvpu (TypeACVPU, optional): Consciousness level based on the ACVPU scale (NEWS2 standard).
    """
    temperature: Optional[float] = Field(None, description="Temperature in Celsius. Convert from F if needed.")
    heartrate: Optional[int] = Field(None, description="Heart rate (BPM).")
    resprate: Optional[int] = Field(None, description="Respiratory rate (BPM).")
    o2sat: Optional[int] = Field(None, description="O2 Saturation (%) on admission.")
    sbp: Optional[int] = Field(None, description="Systolic BP (mmHg).")
    dbp: Optional[int] = Field(None, description="Diastolic BP (mmHg).")
    pain: Optional[int] = Field(None, description="Pain score (0-10).")
    acuity: Optional[int] = Field(None, description="ESI Acuity Level (1-5) if stated or clearly inferable.")
    supplemental_oxygen: Optional[bool] = Field(None, description="True if on O2, False if Room Air.")
    gcs: Optional[int] = Field(None, description="Escala de Coma de Glasgow (3-15)")
    avpu: Optional[TypeAVPU] = Field(None, description="Level of consciousness following MEWS standard. Use 'Alert' only if fully oriented")
    acvpu: Optional[TypeACVPU] = Field(
        None, 
        description="Level of consciousness following NEWS2 standard. Use 'Confusion' for new onset confusion, delirium, or disorientation. Use 'Alert' only if fully oriented."
    )



class LabsSchema(BaseModel):
    """
    Schema for key laboratory test results. 
    Focuses on critical values required for standard risk scores (MEWS, NEWS2, etc.).
    
    Attributes:
        potassium (float, optional): Serum Potassium level in mEq/L.
        sodium (float, optional): Serum Sodium level in mEq/L.
        creatinine (float, optional): Serum Creatinine level in mg/dL.
        wbc (float, optional): White Blood Cell count in K/uL.
        platelets (float, optional): Platelet count in K/uL.
        inr (float, optional): International Normalized Ratio (coagulation).
        albumin (float, optional): Serum Albumin in g/dL.
        bilirubin (float, optional): Total Bilirubin in mg/dL.
        ast (float, optional): Aspartate Aminotransferase level in U/L.
        alt (float, optional): Alanine Aminotransferase level in U/L.
    """
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
    """
    Schema for synthesizing structured physical examination findings.
    
    Attributes:
        abdomen (str, optional): Key findings from the abdominal exam (e.g., tenderness, distension).
        extremities (str, optional): Findings related to extremities (e.g., edema, pulse deficits).
        mental_status (str, optional): Narrative description of the patient's mental status or neurological state.
    """
    abdomen: Optional[str] = Field(None, description="Specific abdominal findings (e.g., 'distended', 'soft').")
    extremities: Optional[str] = Field(None, description="Edema, pulses, or perfusion findings.")
    mental_status: Optional[str] = Field(None, description="Detailed mental status exam notes.")

class ClinicalSchema(BaseModel):
    """
    Aggregated schema that bundles all objective clinical data (Vitals, Labs, Exam Findings).
    
    Attributes:
        vitals (VitalsSchema): Structured vital signs.
        labs (LabsSchema): Structured laboratory results.
        findings (FindingsSchema): Structured physical exam findings.
    """
    vitals: VitalsSchema = Field(..., description="Admission vital signs.")
    labs: LabsSchema = Field(..., description="Pertinent lab results.")
    findings: FindingsSchema = Field(..., description="Physical exam highlights.")

class DrugSchema(BaseModel):
    """
    Schema representing a single medication entry.
    
    Attributes:
        drug (str): The generic name of the medication.
        dose (str, optional): The dosage strength and unit (e.g., '500 mg').
        route (str, optional): The administration route (e.g., 'PO', 'IV', 'SC').
        frequency (str, optional): The frequency of administration (e.g., 'BID', 'Q4H').
    """
    drug: str = Field(..., description="Generic name of the medication.")
    dose: Optional[str] = Field(None, description="Dose with units (e.g., '40 mg').")
    route: Optional[str] = Field(None, description="Route (PO, IV, IM).")
    frequency: Optional[str] = Field(None, description="Frequency (e.g., 'DAILY', 'BID').")


class DeltaSchema(BaseModel):
    """
    Schema for tracking changes in medication regimens (Medication Reconciliation).
    
    Attributes:
        drug (str): Name of the medication being modified.
        change_type (TypeChange): The nature of the change (e.g., NEW_START, STOPPED).
        from_dose (str, optional): The original dose before the change.
        to_dose (str, optional): The new dose after the change.
        reason_inferred (str, optional): Clinical reasoning derived from the text associated with the change.
    """
    drug: str = Field(..., description="Name of the drug changed.")
    change_type: TypeChange = Field(..., description="Nature of the change.")
    from_dose: Optional[str] = Field(None, description="Dose prior to admission/change.")
    to_dose: Optional[str] = Field(None, description="Dose at discharge.")
    reason_inferred: Optional[str] = Field(None, description="Inferred clinical reasoning for the change.")

class TreatmentsSchema(BaseModel):
    """
    Comprehensive schema for medication management and reconciliation.
    
    Attributes:
        admission_drugs (List[DrugSchema]): List of medications the patient was taking prior to admission.
        discharge_drugs (List[DrugSchema]): List of medications prescribed at discharge.
        delta_analysis (List[DeltaSchema]): Structured analysis of changes between admission and discharge medications.
    """
    admission_drugs: List[DrugSchema] = Field(default_factory=list, description="Home meds or meds on arrival.")
    discharge_drugs: List[DrugSchema] = Field(default_factory=list, description="Prescriptions at discharge.")
    delta_analysis: List[DeltaSchema] = Field(default_factory=list, description="Reconciliation of changes.")

class ScribeSchema(BaseModel):
    """
    Master Output Schema for the Scribe Agent.
    This serves as the root structure for all extracted and synthesized clinical data from the TriaLogic pipeline.
    
    Attributes:
        metadata (MetadataSchema): Administrative and classification data.
        semantics (SemanticSchema): High-level clinical summary and context.
        clinical (ClinicalSchema): Structured objective data (vitals, labs, physical exam).
        treatments (TreatmentsSchema): Medication reconciliation and therapy details.
    """
    metadata: MetadataSchema
    semantics: SemanticSchema
    clinical: ClinicalSchema
    treatments: TreatmentsSchema
