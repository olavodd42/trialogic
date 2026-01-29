from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator, PrivateAttr
import logging
import re

logger = logging.getLogger(__name__)

TypeACVPU = Literal['Alert', 'Confusion', 'Voice', 'Pain', 'Unresponsive']
TypeVitalFormat = Literal['LABELED', 'UNLABELED_SEQUENCE', 'MIXED', 'NOT_FOUND']
PHYSICAL_EXAM_PATTERNS = [
    r'\bCARDIAC\b',
    r'\bRRR\b',
    r'\bS1\b',
    r'\bS2\b',
    r'\bMURMUR\b',
    r'\bCTAB\b',
    r'\bLUNGS?\b',
    r'\bRESP\b',
    r'\bABDOMEN\b',
    r'\bSOFT\b',
    r'\bNTND\b',
    r'\bNEURO\b',
    r'\bPERRL\b',
    r'\bEOMI\b',
    r'\bMAE\b',
    r'\bNO DRIFT\b',
    r'\bFOLLOWS COMMANDS\b'
]

VITAL_LABEL_PATTERNS = [
    r'\bBP\b',
    r'\bHR\b',
    r'\bRR\b',
    r'\bTEMP\b',
    r'\bT\b',
    r'\bSPO2\b',
    r'\bO2\b',
    r'\bVITALS?\b',
    r'\bVS\b'
]

def is_physical_exam(span: str) -> bool:
    if not span:
        return False

    span_upper = span.upper()

    # 1️⃣ Se contém termos clássicos de exame físico
    if any(re.search(p, span_upper) for p in PHYSICAL_EXAM_PATTERNS):
        # 2️⃣ …e NÃO contém nenhum rótulo de sinal vital
        if not any(re.search(p, span_upper) for p in VITAL_LABEL_PATTERNS):
            return True

    return False

def looks_like_sequence(span: str) -> bool:
    if not span:
        return False
    nums = re.findall(r'\d+(?:\.\d+)?', span)
    return len(nums) >= 4

class VitalsSchema(BaseModel):
    """
    Standardized schema for vital signs extractor.
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
    reasoning: str = Field(
        ..., 
        description="Brief logic: Where are the vitals located?"
    )

    vital_section_span: Optional[str] = Field(
        ..., 
        description="Exact text span containing vitals (must contain numbers'"
    )

    span_format: TypeVitalFormat = Field(
        ..., 
        description="LABELED, UNLABELED_SEQUENCE, MIXED, NOT_FOUND"
    )

    heartrate: Optional[int] = Field(None, description="Heart Rate (BPM)")
    resprate: Optional[int] = Field(None, description="Respiratory Rate (breaths/min)")
    temperature: Optional[float] = Field(None, description="Temperature (F or C)")
    o2sat: Optional[int] = Field(None, description="Oxygen Saturation (%).")
    sbp: Optional[int] = Field(None, description="Systolic BP")
    dbp: Optional[int] = Field(None, description="Diastolic BP")
    
    supplemental_oxygen: bool = Field(False, description="True if 'NC','mask','L/min' etc.")
    acvpu: TypeACVPU = Field(..., description="Mental status: Alert, Confusion, Voice, Pain, Unresponsive")
    _used_numeric_values: set[float] = PrivateAttr(default_factory=set)

    # ---------------- UTILITIES ----------------
    def _parse_range_or_val(self, val_str: str) -> float:
        clean = re.sub(r'[^\\d.-]', '', val_str)
        if '-' in clean:
            parts = [float(x) for x in clean.split('-') if x.strip()]
            return sum(parts) / len(parts) if parts else 0.0
        try:
          return float(clean)
        except ValueError:
            return 0.0


    def _consume(self, val: float) -> bool:
        if val in self._used_numeric_values:
         return False
        self._used_numeric_values.add(val)
        return True
    
    
    # ---------------- GUARDRAILS ----------------
    @model_validator(mode="after")
    def reject_physical_exam(self):
        if self.vital_section_span and is_physical_exam(self.vital_section_span):
            logger.warning("Physical exam detected. Rejecting span.")
            self.vital_section_span = None
            self.span_format = 'NOT_FOUND'
        return self
                
    # ---------------- RECOVERY ----------------
    @model_validator(mode="after")
    def recover_labeled_and_sequence(self):
        span = self.vital_section_span


        if not span:
            return self
        

        is_sequence = looks_like_sequence(span)
        
        if is_sequence:
            self.span_format = 'UNLABELED_SEQUENCE'

        span_up = span.upper()
        bp_match = re.search(r'(\d{2,3})\s*[/-]\s*(\d{2,3})', span)
        if bp_match:
            try:
                sbp = float(bp_match.group(1))
                dbp = float(bp_match.group(2))
                if 40 < sbp < 300 and 30 < dbp < 200:
                    self.sbp = int(sbp)
                    self.dbp = int(dbp)
            except Exception:
                logger.debug("BP parse failed in recover_labeled_and_sequence", exc_info=True)

        o2_match = re.search(
            r'(?i)\b(?:SAO2|SPO2|O2SAT|O2)\s*[:=]?\s*(\d{2,3})\s*%?',
            span
        )

        if not o2_match:
            # fallback: naked percentage like "94%"
            o2_match = re.search(r'\b(\d{2,3})\s*%', span)

        if o2_match:
            try:
                o2_val = int(o2_match.group(1))
                if 50 <= o2_val <= 100:
                    self.o2sat = o2_val
            except Exception:
                pass

        span_wo_bp = re.sub(r'\d{2,3}\s*[/-]\s*\d{2,3}', ' ', span)
        if not is_sequence:
            parts = re.split(r'[|,\n]', span)
            for part in parts:
                up = part.upper()
                nums = re.findall(r"\d+(?:\.\d+)?", part)


                for n in nums:
                    val = float(n)


                    # Temperature
                    if self.temperature is None and any(re.search(r'\b(TEMP|TEMPERATURE|T[:\s])', up) for _ in [0]):
                        if nums:
                            tval = self._parse_range_or_val(nums[0])
                            if 35 <= tval <= 105:
                                # consume and set
                                if self._consume(tval):
                                    self.temperature = tval
                                    continue


                    # HR
                    if self.heartrate is None and 'HR' in up or 'PULSE' in up or re.search(r'\bP[:\s]\b', up):
                        if nums:
                            hval = self._parse_range_or_val(nums[0])
                            if 30 <= hval <= 250 and self._consume(hval):
                                self.heartrate = int(round(hval))
                                continue

                    # RR
                    if self.resprate is None and ('RR' in up or 'RESP' in up) and 8 <= val <= 40:
                        if nums:
                            rval = self._parse_range_or_val(nums[0])
                            if 5 <= rval <= 60 and self._consume(rval):
                                self.resprate = int(round(rval))
                                continue

                    # O2 Sat
                    if self.o2sat is None and ('%' in part) or ('O2' in up) or ('SPO2' in up) or ('SAT' in up):
                        if nums:
                            oval = self._parse_range_or_val(nums[0])
                            if 50 <= oval <= 100 and self._consume(oval):
                                self.o2sat = int(round(oval))
                                continue

                    for n in nums:
                        val = self._parse_range_or_val(n)
                        if val == 0:
                            continue

                        if self.temperature is None and ((35 <= val <= 42) or (95 <= val <= 105)):
                            if self._consume(val):
                                self.temperature = val
                                break

                        if self.heartrate is None and (40 <= val <= 180):
                            if self._consume(val):
                                self.heartrate = int(round(val))
                                break

                        if self.resprate is None and (8 <= val <= 40):
                            if self._consume(val):
                                self.resprate = int(round(val))
                                break
                            
                        if self.o2sat is None and '%' in part:
                            if 50 <= val <= 100 and self._consume(val):
                                self.o2sat = int(round(val))
                                break
            return self
        
        vals = []
        for n in re.findall(r"\d+(?:\.\d+)?", span_wo_bp):
            try:
                vals.append(float(n))
            except Exception:
                pass

        if self.temperature is None and vals:
            candidate = vals[0]
            if (candidate != int(candidate)) or (35 <= candidate <= 42) or (95 <= candidate <= 105):
                self.temperature = candidate
                vals.pop(0)

        if self.heartrate is None and vals:
            candidate = vals.pop(0)
            if 30 <= candidate <= 250:
                self.heartrate = int(round(candidate))

        if self.resprate is None and vals:
            candidate = vals.pop(0)
            if 5 <= candidate <= 60:
                self.resprate = int(round(candidate))

        # if self.o2sat is None and vals and '%' in span:
        #     candidate = vals.pop(0)
        #     if 50 <= candidate <= 100:
        #         self.o2sat = int(round(candidate))

        return self
    
    @model_validator(mode="after")
    def enforce_label_binding(self):
        """
        Strong regex-based corrections: if explicit labels are present,
        prefer regex-captured values over heuristics.
        """
        span = self.vital_section_span
        if not span or self.span_format == 'NOT_FOUND':
            return self

        clean_span = span.replace('\n', ' ').strip()

        patterns = {
            'heartrate': r'(?i)\b(?:HR|PULSE|P)[:\s=-]*?(\d{2,3})\b',
            'resprate': r'(?i)\b(?:RR|RESP|R)[:\s=-]*?(\d{1,2})\b',
            'temperature': r'(?i)\b(?:T|TEMP|TEMPERATURE)[:\s=-]*?(\d{2,3}\.?\d?)\b',
            'o2sat': r'(?i)\b(?:O2|SAT|SPO2|O2SAT)[:\s=-]*?(\d{2,3})\b'
        }

        # HR
        m = re.search(patterns['heartrate'], clean_span)
        if m:
            try:
                hr_val = int(m.group(1))
                if 30 <= hr_val <= 250:
                    if self.heartrate != hr_val:
                        logger.warning(f"AUTO-CORRECTION: Overriding HR {self.heartrate} -> {hr_val}")
                    self.heartrate = hr_val
            except Exception:
                pass

        # RR
        m = re.search(patterns['resprate'], clean_span)
        if m:
            try:
                rr_val = int(m.group(1))
                if 3 <= rr_val <= 60:
                    self.resprate = rr_val
            except Exception:
                pass

        # Temp
        m = re.search(patterns['temperature'], clean_span)
        if m:
            try:
                t_val = float(m.group(1))
                if 25 <= t_val <= 105:  # broad guard
                    if self.temperature is None or abs(self.temperature - t_val) > 0.5:
                        logger.warning(f"AUTO-CORRECTION: Overriding Temp {self.temperature} -> {t_val}")
                    self.temperature = t_val
            except Exception:
                pass

        # O2
        m = re.search(patterns['o2sat'], clean_span)
        if m:
            try:
                o2_val = int(m.group(1))
                if 0 <= o2_val <= 100:
                    self.o2sat = o2_val
            except Exception:
                pass

        return self

    @model_validator(mode='after')
    def clinical_sanity_check(self):
        logger.debug("Checking clinical sanity...")
        if self.sbp and self.heartrate:
            if abs(self.sbp - self.heartrate) <= 3:
                logger.warning("SBP≈HR collision. Resetting HR.")
                self.heartrate = None
        
        if self.o2sat and self.o2sat > 100:
            self.o2sat = None

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
        
        return self
    
class ClinicalSchema(BaseModel):
    chief_complaint: str = Field(..., description="Main reason for admission (brief)")
    vitals: VitalsSchema
