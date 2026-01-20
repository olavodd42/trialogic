from typing import List, Optional, Literal, Union, Any
from pydantic import BaseModel, Field, model_validator
import logging
import re

# Logger para auditoria (Essencial para TCC e Debugging)
logger = logging.getLogger(__name__)

# --- Enums e Tipos ---
TypeAdmissionType = Literal['EMERGENCY', 'URGENT', 'ELECTIVE']
TypeService = Literal['MEDICINE', 'SURGERY', 'ORTHOPEDICS', 'NEUROLOGY', 'PSYCHIATRY', 'OTHER']
TypeAVPU = Literal['Alert', 'Voice', 'Pain', 'Unresponsive']
TypeACVPU = Literal['Alert', 'Confusion', 'Voice', 'Pain', 'Unresponsive']
TypeChange = Literal['NEW_START', 'DOSE_INCREASE', 'DOSE_DECREASE', 'STOPPED', 'UNCHANGED']
TypeVitalFormat = Literal['LABELED', 'UNLABELED_SEQUENCE', 'MIXED', 'NOT_FOUND']

class VitalsSchema(BaseModel):
    """
    Modelo de domínio para Vitais (Output final limpo).
    """
    heartrate: Optional[int] = None
    resprate: Optional[int] = None
    temperature: Optional[float] = None
    o2sat: Optional[int] = None
    sbp: Optional[int] = None
    dbp: Optional[int] = None
    avpu: Optional[str] = None
    acvpu: Optional[TypeACVPU] = None
    supplemental_oxygen: bool = Field(default=False)
    acuity: Optional[int] = None

class RawVitalsLLM(BaseModel):
    """
    Schema Intermediário para Extração via LLM.
    Padrão: CoT (Chain of Thought) + Grounding.
    """
    # 1. Passo de Raciocínio (Tech Lead: Isso aumenta a precisão drasticamente)
    reasoning: str = Field(
        ..., 
        description="Brief logic: Where are the vitals located? Is it a list or a sentence?"
    )
    
    # 2. Ancoragem (Grounding)
    vital_section_span: Optional[str] = Field(
        ..., 
        description="EXACT copy of the text segment containing vital signs. Example: 'VS: T 99.5, BP 160/81...'"
    )

    span_format: TypeVitalFormat = Field(
        ..., 
        description="STRICT RULE: 'LABELED' requires specific keys NEXT TO values (e.g. 'HR: 80', 'BP 120/80'). Generic headers like 'VS:', 'Vitals:', or units like 'RA', '%', 'F' DO NOT count as labels. If it's just numbers with a header, use 'UNLABELED_SEQUENCE'."
    )

    # 3. Extração Estruturada (Tentativa do Modelo)
    heartrate: Optional[int] = Field(None, description="Heart Rate (BPM)")
    resprate: Optional[int] = Field(None, description="Respiratory Rate (breaths/min)")
    temperature: Optional[float] = Field(None, description="Temperature (F or C)")
    o2sat: Optional[int] = Field(None, description="Oxygen Saturation (%). Generally accompanied by '%'.")
    sbp: Optional[int] = Field(None, description="Systolic BP")
    dbp: Optional[int] = Field(None, description="Diastolic BP")
    
    # 4. Flags de Contexto
    supplemental_oxygen: bool = Field(False, description="True if text mentions 'NC', 'mask', 'L/min'. False if 'RA', 'Room Air'.")
    acvpu: TypeACVPU = Field(..., description="Mental status: Alert, Confusion, Voice, Pain, Unresponsive")

    @model_validator(mode='after')
    def clinical_sanity_check(self):
        """
        Guardrails Clínicos (Padrão Ouro).
        """
        # SBP vs HR Confusion
        if self.sbp and self.heartrate and self.sbp == self.heartrate:
            logger.warning(f"GUARDRAIL: SBP ({self.sbp}) == HR. Assumindo erro de parsing. Resetando HR.")
            self.heartrate = None
        
        # Impossible O2
        if self.o2sat and self.o2sat > 100:
            self.o2sat = None
            
        return self
    
    @model_validator(mode="after")
    def unlabeled_sequence_o2_override(self):
        if (
            self.span_format == "UNLABELED_SEQUENCE"
            and self.vital_section_span
        ):
            nums = re.findall(r'\b\d{2,3}\b', self.vital_section_span)

            # Temp, HR, RR já consumiram posições anteriores
            # Último número = SpO2
            if nums:
                last = int(nums[-1])

                if self.o2sat != last:
                    logger.warning(
                        f"O2 POSITION OVERRIDE: LLM={self.o2sat}, POS={last}"
                    )
                    self.o2sat = last

        return self

class RawScribeLLM(BaseModel):
    chief_complaint: Optional[str]
    vitals: RawVitalsLLM
    # labeled_vitals: Optional[dict] = None
    # unlabeled_vital_sequence: Optional[str] = None
    # supplemental_oxygen: Optional[bool]
    # acvpu: Optional[TypeACVPU]

class ClinicalSchema(BaseModel):
    chief_complaint: str = Field(..., description="Main reason for admission (brief)")
    vitals: VitalsSchema
