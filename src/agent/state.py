# src/agent/state.py
from typing import TypedDict, List, Optional, Annotated
import operator
from langchain_core.messages import BaseMessage

# Define a estrutura de dados para o Score de Risco (Ex: NEWS/MEWS)
class RiskScore(TypedDict):
    score_name: str      # Ex: "NEWS"
    value: int           # Ex: 5
    risk_level: str      # Ex: "Médio"
    clinical_rationale: str # O PORQUÊ (Crucial para a tese)

# O Estado Global do Agente
class TriageState(TypedDict):
    # O histórico da conversa (input do utilizador + respostas do agente)
    # operator.add garante que as novas mensagens são adicionadas e não sobrescritas
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Contexto do Paciente (Recuperado do Banco de Dados)
    subject_id: Optional[int]
    stay_id: Optional[int]
    
    # Dados Clínicos Brutos (Texto recuperado do RAG/SQL)
    clinical_context: Optional[str]
    
    # Output Estruturado (O objetivo final do agente)
    final_risk_assessment: Optional[RiskScore]
    
    # Flags de controlo de fluxo (Para o Grafo decidir o próximo passo)
    missing_data: bool