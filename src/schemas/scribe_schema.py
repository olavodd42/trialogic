from typing import List, Optional, Literal, Dict, Tuple
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
    acvpu: TypeACVPU = "Alert"
    supplemental_oxygen: bool = Field(default=False)

class ClinicalSchema(BaseModel):
    chief_complaint: str = Field(..., description="Main reason for admission (brief)")
    vitals: VitalsSchema

class RawScribeOutput(BaseModel):
    chief_complaint: str = Field(
        ..., 
        description="The main reason for the patient's visit or emergency. Be concise."
    )
    
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
    acvpu: TypeACVPU = Field("Alert", description="Mental status: Alert, Confusion, Voice, Pain, Unresponsive")
    _used_numeric_values: set[float] = PrivateAttr(default_factory=set)
    _assigned_indices: set[int] = PrivateAttr(default_factory=set)

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
    
    def _find_numbers_with_positions(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Returns list of (start, end, token_str) for numeric tokens and percentages.
        """
        nums = []
        # capture numbers with optional trailing %
        for m in re.finditer(r'\d+(?:\.\d+)?\s*%?', text):
            nums.append((m.start(), m.end(), m.group(0).strip()))
        return nums
 
    def _label_patterns(self) -> Dict[str, List[str]]:
        """
        Map logical field -> list of regex label patterns (case-insensitive).
        Keep patterns expressive, prefer T:, HR:, P:, etc.
        """
        return {
            'temperature': [r'(?i)\bTEMP\b', r'(?i)\bTEMPERATURE\b', r'(?i)\bT[:\s]'],
            'heartrate': [r'(?i)\bHR\b', r'(?i)\bHEART RATE\b', r'(?i)\bPULSE\b', r'(?i)\bP[:\s]'],
            'resprate': [r'(?i)\bRR\b', r'(?i)\bRESP\b', r'(?i)\bR[:\s]'],
            'o2sat': [r'(?i)\bSPO2\b', r'(?i)\bSAO2\b', r'(?i)\bO2SAT\b', r'(?i)\bO2\b', r'(?i)\bSAT\b'],
            # BP handled separately via / regex
        }

    def _bind_label_to_nearest_number(self, span: str, max_distance_chars: int = 30) -> None:
        """
        Find label occurrences in span and bind them to the closest numeric token
        within max_distance_chars characters. Mark indices as consumed. This
        sets self.temperature, self.heartrate, self.resprate, self.o2sat when found.
        """
        if not span:
            return

        text = span
        numbers = self._find_numbers_with_positions(text)  # list of (start,end,token)
        if not numbers:
            return

        # Build map of token positions to parsed float
        num_pos_vals: List[Tuple[int, int, float, str]] = []
        for s, e, tok in numbers:
            try:
                # strip trailing % when parsing numeric value
                num_val = float(re.sub(r'[%\s]+', '', tok))
                num_pos_vals.append((s, e, num_val, tok))
            except Exception:
                continue

        if not num_pos_vals:
            return

        label_map = self._label_patterns()
        # For each label type, find occurrences and bind nearest numeric token
        for field, pats in label_map.items():
            for pat in pats:
                for lm in re.finditer(pat, text):
                    lstart = lm.start()
                    # find nearest numeric token not already assigned (by char index)
                    best = None
                    best_dist = None
                    for (ns, ne, val, rawtok) in num_pos_vals:
                        # skip if numeric token region overlaps already assigned index set
                        if any((i >= ns and i < ne) for i in self._assigned_indices):
                            continue
                        # compute char distance between label start and numeric start
                        dist = abs(ns - lstart)
                        if dist <= max_distance_chars:
                            if best is None or (best_dist is not None and dist < best_dist):
                                best = (ns, ne, val, rawtok)
                                best_dist = dist
                    if best:
                        ns, ne, val, rawtok = best
                        assigned = False
                        try:
                            if field == 'temperature':
                                # accept if plausible temperature (C or F)
                                if (35 <= val <= 45) or (95 <= val <= 110):
                                    self.temperature = val
                                    assigned = True
                            elif field == 'heartrate':
                                if 25 <= val <= 300:
                                    self.heartrate = int(round(val))
                                    assigned = True
                            elif field == 'resprate':
                                if 3 <= val <= 60:
                                    self.resprate = int(round(val))
                                    assigned = True
                            elif field == 'o2sat':
                                if 0 <= val <= 100:
                                    self.o2sat = int(round(val))
                                    assigned = True
                        except Exception:
                            assigned = False

                        if assigned:
                            # mark chars as consumed to avoid reuse
                            for i in range(ns, ne):
                                self._assigned_indices.add(i)


    def _map_pipe_sequence(self, span: str):
        """
        High-confidence mapping for pipe/VS style spans:
        expected order: Temp | BP | HR | RR | O2
        Returns nothing but fills self.temperature, self.sbp,self.dbp,self.heartrate,self.resprate,self.o2sat when found.
        """
        if not span:
            return
        s = span.strip()
        # split preferentially by pipe, then comma, else by multiple spaces
        if '|' in s:
            fields = [f.strip() for f in s.split('|') if f.strip()]
        elif ',' in s and len(re.findall(r'\d+(?:\.\d+)?', s)) >= 4:
            fields = [f.strip() for f in s.split(',') if f.strip()]
        else:
            # fallback: split on two-or-more spaces (common in some notes)
            fields = [f.strip() for f in re.split(r'\s{2,}', s) if f.strip()]

        # helper to parse first numeric in a field
        def first_num(f):
            m = re.search(r'\d+(?:\.\d+)?', f)
            return float(m.group(0)) if m else None

        # Temperature candidate
        if self.temperature is None and fields:
            t = first_num(fields[0])
            if t is not None and ((t != int(t)) or (35 <= t <= 45) or (95 <= t <= 110)):
                self.temperature = t

        # BP search across fields (some fields are '120/80' possibly in any position)
        for f in fields:
            m = re.search(r'(\d{2,3})\s*[/-]\s*(\d{2,3})', f)
            if m:
                sbp = int(m.group(1)); dbp = int(m.group(2))
                if 40 < sbp < 300 and 30 < dbp < 200:
                    self.sbp = sbp; self.dbp = dbp
                    break

        # Now assign HR/RR/O2 by conventional positions: HR is field after BP (or the next numeric)
        # Build flattened numeric list in field order excluding BP parts
        flat = []
        for f in fields:
            # exclude the bp token inside field
            f_clean = re.sub(r'\d{2,3}\s*[/-]\s*\d{2,3}', ' ', f)
            toks = re.findall(r'\d+(?:\.\d+)?', f_clean)
            for t in toks:
                flat.append(float(t))

        # remove temp if assigned (first numeric)
        if self.temperature is not None and flat:
            try:
                flat.remove(float(self.temperature))
            except ValueError:
                pass

        # remove sbp/dbp if present in flat
        if self.sbp is not None and self.dbp is not None:
            try:
                flat.remove(float(self.sbp))
            except ValueError:
                pass
            try:
                flat.remove(float(self.dbp))
            except ValueError:
                pass

        # Assign HR then RR then O2 (if present)
        if not self.heartrate and flat:
            v = flat.pop(0)
            if 25 <= v <= 300:
                self.heartrate = int(round(v))
        if not self.resprate and flat:
            v = flat.pop(0)
            if 3 <= v <= 60:
                self.resprate = int(round(v))
        if not self.o2sat:
            # look for percent in original span
            m = re.search(r'(\d{1,3})\s*%(\s*RA|/RA|\b)?', span, re.I)
            if m:
                o2 = int(m.group(1))
                if 0 <= o2 <= 100:
                    self.o2sat = o2
            elif flat:
                cand = flat.pop(0)
                if 0 <= cand <= 100:
                    self.o2sat = int(round(cand))

    # ---------------- GUARDRAILS ----------------
    @model_validator(mode="after")
    def reject_physical_exam(self):
        if self.vital_section_span and is_physical_exam(self.vital_section_span):
            logger.warning("Physical exam detected. Rejecting span.")
            self.vital_section_span = None
            for v in (self.heartrate, self.resprate, self.temperature, self.sbp, self.o2sat):
                v = None

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

                    for m in re.finditer(r'\d{2,3}\s*[/-]\s*\d{2,3}', span):
                        for i in range(m.start(), m.end()):
                            self._assigned_indices.add(i)
            except Exception:
                logger.debug("BP parse failed in recover_labeled_and_sequence", exc_info=True)

        o2_match = re.search(r'(?i)\b(?:saO2|sao2|spo2|o2sat|o2|sat)\b[:=]?\s*(\d{1,3})(?:\s*%|/ra|/RA)?', span)
        if not o2_match:
            o2_match = re.search(r'(\d{1,3})\s*(?:%|/ra|RA\b)', span, re.I)

        if o2_match:
            try:
                o2_val = int(o2_match.group(1))
                if o2_match:
                    o2_val = int(o2_match.group(1))
                    if 0 <= o2_val <= 100:
                        self.o2sat = o2_val
                # if 0 <= o2_val <= 100:
                #     self.o2sat = o2_val
                #     for m in re.finditer(re.escape(o2_match.group(0)), span, re.IGNORECASE):
                #         for i in range(m.start(), m.end()):
                #             self._assigned_indices.add(i)

            except Exception:
                pass

        nums_tokens = re.findall(r'\d+(?:\.\d+)?', span)
        nums = [float(t) for t in nums_tokens]
        def _remove_first(queue, value):
            for i, v in enumerate(queue):
                if abs(v - value) < 0.001 or round(v) == round(value):
                    queue.pop(i)
                    return True
            return False

        if self.sbp is not None and self.dbp is not None:
            _remove_first(nums, float(self.sbp))
            _remove_first(nums, float(self.dbp))
        if self.o2sat is not None:
            _remove_first(nums, float(self.o2sat))
        
        self._bind_label_to_nearest_number(span, max_distance_chars=30)
        # span_wo_bp = re.sub(r'\d{2,3}\s*[/-]\s*\d{2,3}', ' ', span)
        if not is_sequence:
            tokens = [float(t) for t in re.findall(r'\d+(?:\.\d+)?', span)]

            if self.sbp is not None and self.dbp is not None:
                _remove_first(tokens, float(self.sbp))
                _remove_first(tokens, float(self.dbp))
            if self.o2sat is not None:
                _remove_first(tokens, float(self.o2sat))
            if self.temperature is not None:
                _remove_first(tokens, float(self.temperature))
            if self.heartrate is not None:
                _remove_first(tokens, float(self.heartrate))

            if self.o2sat is None and ('%' in span_up or 'O2' in span_up or 'SAT' in span_up):
                if nums:
                    candidate = nums.pop(0)
                    if 0 <= candidate <= 100:
                        self.o2sat = int(round(candidate))

            if self.vital_section_span is None:
                for v in (self.heartrate, self.resprate, self.temperature, self.sbp, self.o2sat, self.dbp):
                    v = None
        
            return self
        
        span = span.strip()
        pipey = ('|' in span) or ('VS' in span_up) or ('VITALS' in span_up)

        
        if pipey:
            self._map_pipe_sequence(span)

        else:
            # Non-pipe sequence: positional queue mapping
            q = nums.copy()

            # Temp: optional first if plausible
            if self.temperature is None and q:
                cand = q[0]
                if (cand != int(cand)) or (35 <= cand <= 45) or (95 <= cand <= 110):
                    self.temperature = cand
                    q.pop(0)

            # Heart rate
            if self.heartrate is None and q:
                cand = q.pop(0)
                if 25 <= cand <= 300:
                    self.heartrate = int(round(cand))
                else:
                    # try next token if first implausible
                    if q:
                        cand2 = q.pop(0)
                        if 25 <= cand2 <= 300:
                            self.heartrate = int(round(cand2))

            # Respiratory rate: crucial step — try immediate token, if implausible scan remaining tokens
            if self.resprate is None:
                if q:
                    cand = q.pop(0)
                    if 3 <= cand <= 60:
                        self.resprate = int(round(cand))
                    else:
                        # scan remaining list for plausible RR
                        assigned = False
                        for i, vv in enumerate(list(q)):
                            if 3 <= vv <= 60:
                                self.resprate = int(round(vv))
                                # remove that token from q
                                for j in range(len(q)):
                                    if abs(q[j] - vv) < 0.001 or round(q[j]) == round(vv):
                                        q.pop(j)
                                        break
                                assigned = True
                                break
                        # if none found, resprate remains None

            # O2 positional fallback if not captured and span contains indicator
            if self.o2sat is None and ('%' in span_up or 'O2' in span_up or 'SAT' in span_up or 'SPO2' in span_up):
                if q:
                    cand = q.pop(0)
                    if 0 <= cand <= 100:
                        self.o2sat = int(round(cand))

        # final: try to fill O2 if still absent by scanning for percent token
        if self.o2sat is None:
            m = re.search(r'(\d{1,3})\s*%', span)
            if m:
                val = int(m.group(1))
                if 0 <= val <= 100:
                    self.o2sat = val

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

        clean = span.replace('\n', ' ').strip()

        patterns = {
            'heartrate': r'(?i)\b(?:HR|HEART RATE|PULSE|P)[:\s=-]*?(\d{1,3})\b',
            'resprate': r'(?i)\b(?:RR|RESP|R)[:\s=-]*?(\d{1,2})\b',
            'temperature': r'(?i)\b(?:T|TEMP|TEMPERATURE)[:\s=-]*?(\d{2,3}\.?\d?)\b',  # require label
            'o2sat': r'(?i)\b(?:SPO2|SAO2|O2SAT|O2|SAT|O2SATS?)[:\s=-]*?(\d{1,3})(?:\s*%|/ra|RA\b)?'
        }

        def _mark_group_chars(m):
            if not m:
                return
            # mark only group 1 numerical characters as assigned
            start, end = m.start(1), m.end(1)
            for i in range(start, end):
                self._assigned_indices.add(i)

        # HR
        m = re.search(patterns['heartrate'], clean)
        if m:
            try:
                hr = int(m.group(1))
                if 20 <= hr <= 300:
                    self.heartrate = hr
                    _mark_group_chars(m)
            except: pass

        # RR
        m = re.search(patterns['resprate'], clean)
        if m:
            try:
                rr = int(m.group(1))
                if 3 <= rr <= 60:
                    self.resprate = rr
                    _mark_group_chars(m)
            except: pass

        # Temp
        m = re.search(patterns['temperature'], clean)
        if m:
            try:
                t = float(m.group(1))
                if (25 <= t <= 45) or (95 <= t <= 110):
                    self.temperature = t
                    _mark_group_chars(m)
            except: pass
            
        # O2
        m = re.search(patterns['o2sat'], clean)
        if m:
            try:
                o2 = int(m.group(1))
                if 0 <= o2 <= 100:
                    self.o2sat = o2
                    _mark_group_chars(m)
            except: pass


        return self
    
    def _map_avpu(self, raw_acvpu: str) -> str:
        """
        Maps the raw ACVPU (Alert, Confusion, Voice, Pain, Unresponsive) scale to the standard AVPU scale.
        
        Business Logic:
        - 'Confusion' is mapped to 'Alert' (as per some triage protocols where confusion is a status of an alert patient, though this can be customized).
        - 'Alert', 'Voice', 'Pain', 'Unresponsive' map to themselves.

        Args:
            raw_acvpu (str): The raw mental status string extracted by the LLM.

        Returns:
            str: The normalized AVPU status. Defaults to 'Alert' if input is empty or unknown.
        """
        if not raw_acvpu:
            return "Alert"
            
        mapping = {
            "Confusion": "Alert",
            "Alert": "Alert",
            "Voice": "Voice",
            "Pain": "Pain",
            "Unresponsive": "Unresponsive"
        }
        return mapping.get(raw_acvpu, "Alert") # Default fallback

    @model_validator(mode='after')
    def clinical_sanity_check(self):
        span = (self.vital_section_span or "").upper()
        if self.temperature is not None:
            # If temperature came from unlabeled heuristics earlier, it may be int like 107
            if float(self.temperature).is_integer() and int(self.temperature) >= 100:
                # find HR explicit label in span
                m_hr_label = re.search(r'(?i)\b(?:HR|PULSE|P)[:\s=-]*?(\d{2,3})\b', span)
                if m_hr_label:
                    hr_val = int(m_hr_label.group(1))
                    if hr_val == int(self.temperature):
                        # prefer HR
                        self.heartrate = hr_val
                        self.temperature = None
                        logger.warning(f"AUTO-CORRECTION: Temperature {hr_val} appears to be HR. Cleared temperature; set HR={hr_val}.")
                        
        if (self.heartrate is None) and span:
            m_hr_label = re.search(r'(?i)\b(?:HR|PULSE|P)[:\s=-]*?(\d{1,3})\b', span)
            if m_hr_label:
                try:
                    hr_val = int(m_hr_label.group(1))
                    if 20 <= hr_val <= 300:
                        self.heartrate = hr_val
                except:
                    pass
                
        if self.heartrate and self.temperature:
            if abs(self.heartrate - round(self.temperature)) <= 1:
                logger.warning("HR≈Temp collision detected. Resetting HR.")
                self.heartrate = None

        if self.sbp and self.heartrate and abs(self.sbp - self.heartrate) <= 3:
            logger.warning("SBP≈HR collision detected. Resetting HR.")
            self.heartrate = None

        if self.o2sat and (self.o2sat < 0 or self.o2sat > 100):
            logger.warning("Invalid O2Sat; clearing.")
            self.o2sat = None

        if self.temperature is not None:
            if not ((25 <= self.temperature <= 45) or (95 <= self.temperature <= 110)):
                logger.warning("Temperature out of plausible ranges; clearing.")
                self.temperature = None

        return self

    def to_domain(self) -> ClinicalSchema:
        """
        Mapper Method: Converte o DTO plano para o modelo hierárquico rico.
        Isso isola a 'sujeira' da extração da 'limpeza' do domínio.
        """
        avpu = self._map_avpu(self.acvpu)
        return ClinicalSchema(
            chief_complaint=self.chief_complaint,
            vitals=VitalsSchema(
                heartrate=self.heartrate,
                sbp=self.sbp,
                dbp=self.dbp,
                resprate=self.resprate,
                temperature=self.temperature,
                o2sat=self.o2sat,
                supplemental_oxygen=self.supplemental_oxygen,
                avpu=avpu,
                acvpu=self.acvpu
            )
        )