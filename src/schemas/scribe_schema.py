from typing import List, Optional, Literal, Union, Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo
import re

# --- Enums e Tipos ---
TypeAdmissionType = Literal['EMERGENCY', 'URGENT', 'ELECTIVE']
TypeService = Literal['MEDICINE', 'SURGERY', 'ORTHOPEDICS', 'NEUROLOGY', 'PSYCHIATRY', 'OTHER']
TypeAVPU = Literal['Alert', 'Voice', 'Pain', 'Unresponsive']
TypeACVPU = Literal['Alert', 'Confusion', 'Voice', 'Pain', 'Unresponsive']
TypeChange = Literal['NEW_START', 'DOSE_INCREASE', 'DOSE_DECREASE', 'STOPPED', 'UNCHANGED']

# --- Helper de Limpeza Nuclear (Regex) ---
def clean_numeric_value(v: Any) -> Optional[float]:
    """
    Função de limpeza agressiva.
    Aceita qualquer lixo que a LLM mandar (ex: "39.2 C", "approx 110", "98% environment")
    e extrai o primeiro número float válido.
    """
    if v is None:
        return None
    
    # Se já vier float/int limpo, retorna logo
    if isinstance(v, (float, int)):
        return float(v)
    
    # Se for string, entra o modo cirurgião
    if isinstance(v, str):
        # Normaliza decimal: troca vírgula por ponto (comum em PT-BR)
        clean_str = v.replace(',', '.')
        
        # Regex que busca: 
        # -?     -> sinal negativo opcional
        # \d+    -> um ou mais dígitos
        # (?:    -> grupo de não-captura para decimal
        # \.\d+  -> ponto seguido de dígitos
        # )?     -> o decimal é opcional
        # Procura o primeiro padrão numérico na string
        match = re.search(r'-?\d+(\.\d+)?', clean_str)
        
        if match:
            try:
                val = float(match.group(0)) # group(0) pega o match inteiro
                # Sanity Check básico
                return val
            except ValueError:
                return None
                
    return None

class MetadataSchema(BaseModel):
    admission_type: Optional[TypeAdmissionType] = Field(None, description="The urgency level of admission.")
    service: Optional[TypeService] = Field(None, description="Primary medical service.")
    discharge_disposition: Optional[str] = Field(None, description="Patient destination after discharge.")

class VitalsSchema(BaseModel):
    """
    Schema de Vitais Blindado e Completo.
    Aceita strings na entrada para permitir formatação suja, limpando com validator.
    """
    
    heartrate: Optional[int] = Field(None, description="Heart rate (HR/BPM).")
    resprate: Optional[int] = Field(None, description="Respiratory Rate (RR) in rpm.")
    temperature: Optional[float] = Field(None, description="Temperature (Temp) in Celsius.  \
    Don't use null here if there is a temperature value in text, even if implicitly.")
    o2sat: Optional[int] = Field(None, description="Oxygen Saturation (SpO2) %.")
    sbp: Optional[int] = Field(None, description="Systolic BP (top number).")
    dbp: Optional[int] = Field(None, description="Diastolic BP (bottom number).")
    pain: Optional[int] = Field(None, description="Pain score (0-10).")
    
    # Campo adicionado conforme seu código
    acuity: Optional[int] = Field(None, description="ESI Acuity Scale (1-5).")
    
    supplemental_oxygen: Optional[bool] = Field(None, description="True if patient is on O2.")
    avpu: Optional[TypeAVPU] = Field(None, description="AVPU scale status.")
    acvpu: Optional[TypeACVPU] = Field(None, description="ACVPU (NEWS2) status.")

    @field_validator('temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', mode='before')
    @classmethod
    def robust_numeric_cleaning(cls, v, info: ValidationInfo):
        # Mantendo o debug comentado para limpeza
        # print(f"DEBUG: Cleaning field '{info.field_name}': val='{v}' type={type(v)}")
        return clean_numeric_value(v)

class LabsSchema(BaseModel):
    """
    Schema para Resultados Laboratoriais (Completo).
    """
    # Campos completos conforme sua solicitação
    potassium: Optional[Union[float, str]] = Field(None, description="Potassium (K+) in mEq/L.")
    sodium: Optional[Union[float, str]] = Field(None, description="Sodium (Na+) in mEq/L.")
    creatinine: Optional[Union[float, str]] = Field(None, description="Creatinine (Cr) in mg/dL.")
    wbc: Optional[Union[float, str]] = Field(None, description="White Blood Cells (WBC) count.")
    platelets: Optional[Union[float, str]] = Field(None, description="Platelets count.")
    inr: Optional[Union[float, str]] = Field(None, description="International Normalized Ratio.")
    albumin: Optional[Union[float, str]] = Field(None, description="Albumin (g/dL).")
    bilirubin: Optional[Union[float, str]] = Field(None, description="Total Bilirubin (mg/dL).")
    ast: Optional[Union[float, str]] = Field(None, description="Aspartate Aminotransferase (U/L).")
    alt: Optional[Union[float, str]] = Field(None, description="Alanine Aminotransferase (U/L).")
    
    @field_validator('potassium', 'sodium', 'creatinine', 'wbc', 'platelets', 'inr', 'albumin', 'bilirubin', 'ast', 'alt', mode='before')
    @classmethod
    def robust_numeric_cleaning(cls, v):
        return clean_numeric_value(v)

class ClinicalSchema(BaseModel):
    chief_complaint: Optional[str] = Field(None, description="Primary patient complaint.")
    vitals: VitalsSchema = Field(..., description="Vital signs measurements.")
    labs: Optional[LabsSchema] = Field(None, description="Laboratory results.")

class SemanticSchema(BaseModel):
    summary: str = Field(..., description="Brief summary of the clinical note.")
    diagnoses: List[str] = Field(default_factory=list, description="List of diagnoses or problems.")

class MedicationSchema(BaseModel):
    name: str = Field(..., description="Name of the medication.")
    dosage: Optional[str] = Field(None, description="Dose and frequency.")
    route: Optional[str] = Field(None, description="Route of administration.")

class DeltaSchema(BaseModel):
    medication: str
    change_type: TypeChange
    reason: Optional[str]

class TreatmentsSchema(BaseModel):
    admission_meds: List[MedicationSchema] = Field(default_factory=list, description="Meds before admission.")
    discharge_meds: List[MedicationSchema] = Field(default_factory=list, description="Meds at discharge.")
    delta_analysis: List[DeltaSchema] = Field(default_factory=list, description="Reconciliation of changes.")

class ScribeSchema(BaseModel):
    metadata: MetadataSchema
    semantics: SemanticSchema
    clinical: ClinicalSchema
    treatments: TreatmentsSchema
    missing_data_justification: Optional[str] = Field(
        None, 
        description="If vitals are missing, briefly explain why."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "clinical": {
                    "vitals": {
                        "temperature": 39.2, 
                        "sbp": 120,
                        "acvpu": "Alert"
                    }
                }
            }
        }