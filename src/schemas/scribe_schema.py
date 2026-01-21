from typing import List, Optional, Literal, Union, Any
from pydantic import BaseModel, Field, model_validator
import logging
import re

logger = logging.getLogger(__name__)

TypeACVPU = Literal['Alert', 'Confusion', 'Voice', 'Pain', 'Unresponsive']
TypeVitalFormat = Literal['LABELED', 'UNLABELED_SEQUENCE', 'MIXED', 'NOT_FOUND']

class VitalsSchema(BaseModel):
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
    reasoning: str = Field(
        ..., 
        description="Brief logic: Where are the vitals located? Is it a list or a sentence?"
    )

    vital_section_span: Optional[str] = Field(
        ..., 
        description="EXACT copy of the text segment containing vital signs. Example: 'VS: T 99.5, BP 160/81...'"
    )

    span_format: TypeVitalFormat = Field(
        ..., 
        description="STRICT RULE: 'LABELED' requires specific keys NEXT TO values (e.g. 'HR: 80', 'BP 120/80'). Generic headers like 'VS:', 'Vitals:', or units like 'RA', '%', 'F' DO NOT count as labels. If it's just numbers with a header, use 'UNLABELED_SEQUENCE'."
    )

    heartrate: Optional[int] = Field(None, description="Heart Rate (BPM)")
    resprate: Optional[int] = Field(None, description="Respiratory Rate (breaths/min)")
    temperature: Optional[float] = Field(None, description="Temperature (F or C)")
    o2sat: Optional[int] = Field(None, description="Oxygen Saturation (%). Generally accompanied by '%'.")
    sbp: Optional[int] = Field(None, description="Systolic BP")
    dbp: Optional[int] = Field(None, description="Diastolic BP")
    
    supplemental_oxygen: bool = Field(False, description="True if text mentions 'NC', 'mask', 'L/min'. False if 'RA', 'Room Air'.")
    acvpu: TypeACVPU = Field(..., description="Mental status: Alert, Confusion, Voice, Pain, Unresponsive")

    def _parse_range_or_val(self, val_str: str) -> float:
        clean = re.sub(r'[^\d.-]', '', val_str)
        if '-' in clean:
            parts = [float(x) for x in clean.split('-') if x.strip()]
            if parts:
                return sum(parts) / len(parts)
        try:
            return float(clean)
        except ValueError:
            return 0.0

    @model_validator(mode='after')
    def clinical_sanity_check(self):
        logger.debug("Checking clinical sanity...")
        if self.sbp and self.heartrate and self.sbp == self.heartrate:
            logger.warning(f"⚠ GUARDRAIL: SBP ({self.sbp}) == HR. Assuming parsing error. Resetting HR.")
            self.heartrate = None
        
        if self.o2sat and self.o2sat > 100:
            self.o2sat = None

        return self
    
    @model_validator(mode="after")
    def clean_blood_pressure(self):
        logger.debug("Cleaning Blood Pressure...")
        if isinstance(self.sbp, str) and '/' in self.sbp:
            try:
                parts = self.sbp.split('/')
                self.sbp = int(parts[0].strip())
                self.dbp = int(parts[1].strip())
            except:
                logger.warning("⚠ Blood Pressure couldn't be parsed.")
                self.sbp = None

        if isinstance(self.sbp, str): 
            logger.warning("⚠ Blood Pressure couldn't be parsed.")
            self.sbp = None
        
        return self

    @model_validator(mode="after")
    def recover_comma_separated(self):
        logger.debug("Checking and correcting vitals...")
        if not self.vital_section_span:
            logger.warning("⚠ No vitals span found.")
            return self
            
        # 1. Dynamic delimiter
        span = self.vital_section_span
        if ',' in span:
            parts = [p.strip() for p in span.split(',') if p.strip()]
        else:
            parts = span.split()
            
        if len(parts) >= 2:
            logger.info(f"Attempting Label-Aware Recovery on span: {parts}")
            
            for part in parts:
                upper_part = part.upper()
                
                # --- 0. Labels Detection ---
                target_field = None
                if any(x in upper_part for x in ['T ', 'TEMP', ' T ']) or upper_part.startswith('T '):
                    target_field = 'temperature'
                elif any(x in upper_part for x in ['P ', 'HR ', 'PULSE', 'HR:', 'P:']):
                    target_field = 'heartrate'
                elif any(x in upper_part for x in ['R ', 'RR ', 'RESP', 'R:', 'RR:']):
                    target_field = 'resprate'
                elif any(x in upper_part for x in ['O2', 'SAT', 'SPO2']):
                    target_field = 'o2sat'
                
                # --- 1. BP Check ---
                if '/' in part: 
                    try:
                        s_str, d_str = part.split('/')
                        s_val = self._parse_range_or_val(s_str)
                        d_val = self._parse_range_or_val(d_str)
                        if s_val > 0 and d_val > 0:
                            self.sbp = int(s_val)
                            self.dbp = int(d_val)
                            continue 
                    except: pass

                # --- 2. Candidates Extraction ---
                candidates = re.findall(r'\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?', part)
                
                for cand in candidates:
                    try:
                        val = self._parse_range_or_val(cand)
                        if val == 0: continue
                    except ValueError: continue

                    if target_field == 'temperature':
                            if (35 <= val <= 105): 
                                self.temperature = val
                                break
                    elif target_field == 'heartrate':
                            if (30 <= val <= 250): 
                                self.heartrate = int(val)
                                break
                    elif target_field == 'resprate':
                            if (8 <= val <= 60): 
                                self.resprate = int(val)
                                break
                    elif target_field == 'o2sat':
                            if (50 <= val <= 100): 
                                self.o2sat = int(val)
                                break
                    
                    if not target_field:
                        # Temp
                        if not self.temperature and ((95 <= val <= 105) or (35 <= val <= 40)):
                            self.temperature = val
                            break 
                        # HR
                        if not self.heartrate and (40 <= val <= 180):
                                if abs(val - round(val)) < 0.1: 
                                    self.heartrate = int(val)
                                    break
                        # RR
                        if not self.resprate and (8 <= val <= 40):
                            if abs(val - round(val)) < 0.1:
                                self.resprate = int(val)
                                break
                        # O2
                        if not self.o2sat and (80 <= val <= 100):
                            if abs(val - round(val)) < 0.1:
                                self.o2sat = int(val)
                                break

        return self

class RawScribeLLM(BaseModel):
    chief_complaint: Optional[str]
    vitals: RawVitalsLLM

    @model_validator(mode="after")
    def validate_clinical_consistency(self):
        logger.debug("Validating Clinical Schema consistency...")
        if not self.chief_complaint or not self.vitals.acvpu:
            return self

        cc_lower = self.chief_complaint.lower()
        acvpu_val = self.vitals.acvpu

        # Regra: Se a queixa é CONFUSÃO, o ACVPU deveria idealmente refletir isso ou ser investigado.
        # "Confusion" no ACVPU é um estado específico.
        if "confusion" in cc_lower or "confused" in cc_lower or "disoriented" in cc_lower:
            if acvpu_val == "Alert":
                logger.warning(
                    f"CLINICAL MISMATCH WARNING: Chief Complaint indicates '{self.chief_complaint}', "
                    f"but ACVPU is '{acvpu_val}'. Verify if patient is Alert but Disoriented."
                )
                # Nota para TCC: Não alteramos o dado automaticamente para evitar 
                # corrupção de dados (pode ser que o paciente melhorou), mas logamos para revisão.
        
        return self
    
class ClinicalSchema(BaseModel):
    chief_complaint: str = Field(..., description="Main reason for admission (brief)")
    vitals: VitalsSchema
