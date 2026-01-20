import logging
import os
from typing import Dict, Any, cast
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from src.schemas.scribe_schema import RawScribeLLM, VitalsSchema, ClinicalSchema
from src.state.agent_state import AgentState
from src.utils.vitals_normalizer import normalize_temperature

# Configuração de Logs
logger = logging.getLogger(__name__)

class ScribeAgent:
    """
    Agente responsável pela extração e estruturação de dados clínicos (Scribe).
    
    Architecture Principle: Single Responsibility Principle (SRP)
    Este agente tem apenas uma razão para mudar: a lógica de como extraímos dados do texto.
    """

    def __init__(self, model: BaseChatModel, prompt_path: str = "prompts/scribe_prompt.md"):
        """
        Injeção de Dependência: O modelo é passado no construtor.
        Isso permite trocar Ollama por OpenAI/Anthropic sem tocar na classe.
        """
        self.model = model
        self.system_prompt = self._load_prompt(prompt_path)
        self.structured_model = self.model.with_structured_output(RawScribeLLM)

    def _load_prompt(self, path: str) -> str:
        """Carrega o prompt do sistema de forma segura."""
        try:
            # Tech Lead Tip: Use caminhos absolutos baseados na raiz do projeto em produção
            full_path = os.path.join(os.getcwd(), path)
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"CRITICAL: Prompt file not found at {path}")
            raise RuntimeError(f"Scribe system prompt missing: {path}")

    def _map_avpu(self, raw_acvpu: str) -> str:
        """
        Business Logic: Mapeamento de ACVPU para AVPU.
        Encapsulado para facilitar testes unitários.
        """
        # Normalização simples
        if not raw_acvpu:
            return "Alert" # Default seguro, mas deve ser auditado
            
        mapping = {
            "Confusion": "Alert",
            "Alert": "Alert",
            "Voice": "Voice",
            "Pain": "Pain",
            "Unresponsive": "Unresponsive"
        }
        return mapping.get(raw_acvpu, "Alert") # Default fallback

    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Executa o pipeline de extração.
        
        Args:
            state: O estado atual do grafo (LangGraph state).
        
        Returns:
            Um dicionário com as atualizações para o estado (updates).
        """
        clinical_text = ""
        input_data = state.get("input")
        
        if input_data:
             # InputSchema is a Pydantic model
             if hasattr(input_data, "raw_text"):
                 clinical_text = input_data.raw_text
             elif isinstance(input_data, dict):
                 clinical_text = input_data.get("raw_text", "")

        # Fallback for compatibility/testing
        if not clinical_text:
             # Using get on TypedDict with unknown key might be flagged, but at runtime it works if dict
             clinical_text = state.get("clinical_text", "") # type: ignore

        if not clinical_text:
            logger.warning("Scribe received empty clinical text.")
            return {"error": "Empty input text"}

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"INPUT NOTES:\n{clinical_text}")
        ]

        try:
            logger.info("Invoking Scribe Model...")
            # A mágica acontece aqui. O Pydantic já valida tipos básicos.
            response: RawScribeLLM = cast(RawScribeLLM, self.structured_model.invoke(messages))
            
            # --- Domain Mapping Layer ---
            # Transformando o DTO (Data Transfer Object) do LLM no Modelo de Domínio
            vitals = response.vitals
            print(f"[DEBUG] VITALS: {vitals}")
            
            # Aplicando lógica de negócios (Ex: AVPU)
            final_acvpu = vitals.acvpu
            avpu_mapped = self._map_avpu(final_acvpu)
            
            domain_vitals = VitalsSchema(
                heartrate=vitals.heartrate,
                resprate=vitals.resprate,
                temperature=normalize_temperature(vitals.temperature) if vitals.temperature else None, # O validator do Pydantic já deve ter normalizado
                o2sat=vitals.o2sat,
                sbp=vitals.sbp,
                dbp=vitals.dbp,
                avpu=avpu_mapped,
                acvpu=final_acvpu,
                supplemental_oxygen=vitals.supplemental_oxygen or False,
                acuity=None 
            )

            clinical_output = ClinicalSchema(
                chief_complaint=response.chief_complaint or "Not reported",
                vitals=domain_vitals
            )
            print(clinical_output)

            logger.info(f"Extraction Successful. HR: {domain_vitals.heartrate}, BP: {domain_vitals.sbp}/{domain_vitals.dbp}")

            return {
                "extracted_data": clinical_output,
                "validation_errors": [], # Limpa erros anteriores se houver retry
                "attempts": state.get("attempts", 0) + 1,
                "is_success": True
            }

        except ValidationError as e:
            # Captura erros de validação do Pydantic que o modelo não conseguiu resolver
            logger.warning(f"Validation Error in Scribe: {str(e)}")
            return {
                "validation_errors": [str(e)],
                "attempts": state.get("attempts", 0) + 1,
                "is_success": False
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in Scribe Agent: {e}", exc_info=True)
            return {
                "error": str(e),
                "is_success": False
            }
