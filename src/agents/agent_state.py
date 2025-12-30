
from typing import Dict, Any, TypedDict, Optional
from src.schemas.scribe_output_schema import ScribeOutputSchema
from src.schemas.input_schema import InputSchema



class AgentState(TypedDict):
    input: InputSchema              # O input bruto da triagem
    extracted_data: Optional[ScribeOutputSchema] # O JSON validado (SinaisVitais)
    validation_error: Optional[str]   # Mensagem de erro do Pydantic (se houver)
    attempts: int