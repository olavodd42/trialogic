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

    def _parse_range_or_val(self, val_str: str) -> float:
        """Helper para calcular média de ranges (ex: '133-123' -> 128.0)"""
        # Remove chars não numéricos exceto ponto e hífen
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
    def clean_blood_pressure(self):
        """
        Única ajuda permitida: Separar 120/80 se a LLM esquecer.
        """
        if isinstance(self.sbp, str) and '/' in self.sbp:
            try:
                parts = self.sbp.split('/')
                self.sbp = int(parts[0].strip())
                self.dbp = int(parts[1].strip())
            except:
                self.sbp = None

        if isinstance(self.sbp, str): 
            self.sbp = None
        
        return self

    @model_validator(mode="after")
    def recover_comma_separated(self):
        """
        FALLBACK TÉCNICO (LABEL & RANGE AWARE - HYBRID SPLIT): 
        Lida com sequências separadas por vírgula OU espaço.
        1. Determina delimitador (vírgula ou espaço).
        2. Procura Rótulos explícitos (T, P, HR, O2).
        3. Se achar rótulo, sobrescreve.
        4. Se não, usa heurística de range.
        """
        if not self.vital_section_span:
            return self
            
        # [TECH LEAD FIX] Delimitador Dinâmico
        # Se tem vírgula, assume CSV. Se não tem, assume Space-Separated.
        span = self.vital_section_span
        if ',' in span:
            parts = [p.strip() for p in span.split(',') if p.strip()]
        else:
            parts = span.split() # Split default por whitespace (espaço, tab, newline)
            
        # Relaxei para >= 2 partes (ex: BP e HR apenas)
        if len(parts) >= 2:
            logger.info(f"Attempting Label-Aware Recovery on span: {parts}")
            
            for part in parts:
                upper_part = part.upper()
                
                # --- 0. Detecção de Rótulos (Prioridade Máxima) ---
                target_field = None
                # Espaços importam para não confundir 'AT' com 'T'
                if any(x in upper_part for x in ['T ', 'TEMP', ' T ']) or upper_part.startswith('T '):
                    target_field = 'temperature'
                elif any(x in upper_part for x in ['P ', 'HR ', 'PULSE', 'HR:', 'P:']):
                    target_field = 'heartrate'
                elif any(x in upper_part for x in ['R ', 'RR ', 'RESP', 'R:', 'RR:']):
                    target_field = 'resprate'
                elif any(x in upper_part for x in ['O2', 'SAT', 'SPO2']):
                    target_field = 'o2sat'
                
                # --- 1. BP Check (Sempre verifica barras) ---
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

                # --- 2. Extração de Candidatos ---
                candidates = re.findall(r'\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?', part)
                
                for cand in candidates:
                    try:
                        val = self._parse_range_or_val(cand)
                        if val == 0: continue
                    except ValueError: continue

                    # SE TEM RÓTULO, FORÇA O VALOR (Override LLM)
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
                    
                    # SE NÃO TEM RÓTULO, USA HEURÍSTICA DE "PREENCHER VAZIOS"
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

    
    # @model_validator(mode="after")
    # def unlabeled_sequence_o2_override(self):
    #     if (
    #         self.span_format == "UNLABELED_SEQUENCE"
    #         and self.vital_section_span
    #     ):
    #         nums = re.findall(r'\b\d{2,3}\b', self.vital_section_span)

    #         # Temp, HR, RR já consumiram posições anteriores
    #         # Último número = SpO2
    #         if nums:
    #             last = int(nums[-1])

    #             if self.o2sat != last:
    #                 logger.warning(
    #                     f"O2 POSITION OVERRIDE: LLM={self.o2sat}, POS={last}"
    #                 )
    #                 self.o2sat = last

    #     return self
    
    # @model_validator(mode="after")
    # def percent_o2_override(self):
    #     if not self.vital_section_span:
    #         return self

    #     match = re.search(r'(\d{2,3})\s*%\s*(RA|NC|NRB|HFNC)?', self.vital_section_span)

    #     if match:
    #         o2 = int(match.group(1))

    #         if self.o2sat != o2:
    #             logger.warning(
    #                 f"O2 PERCENT OVERRIDE: LLM={self.o2sat}, OBSERVED={o2}"
    #             )
    #             self.o2sat = o2

    #         # oxigênio suplementar
    #         if match.group(2) and match.group(2) != "RA":
    #             self.supplemental_oxygen = True
    #         else:
    #             self.supplemental_oxygen = False

    #     return self


class RawScribeLLM(BaseModel):
    chief_complaint: Optional[str]
    vitals: RawVitalsLLM
class ClinicalSchema(BaseModel):
    chief_complaint: str = Field(..., description="Main reason for admission (brief)")
    vitals: VitalsSchema
