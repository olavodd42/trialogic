import os
import logging
import json
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
from dotenv import load_dotenv

from src.state.agent_state import AgentState
from src.schemas.scribe_schema import VitalsSchema
# Importamos a ferramenta, mas vamos usá-la como função Python normal!
from src.tools.calculator import calculate_clinical_score 
from src.schemas.mathematician_schema import MathematicianSchema

load_dotenv()

# Logger Configuration
logger = logging.getLogger(__name__)

SEED = 42

# 1. Configuração do LLM (Sem bind_tools!)
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.1",
    temperature=0,
    seed=SEED
)

# Forçamos uma saída estruturada para o relatório final
mathematician_model = llm.with_structured_output(MathematicianSchema)

# Carregamento robusto do prompt
prompt_path = os.path.join(os.getcwd(), "prompts", "mathematician_prompt.md")
try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        MATHEMATICIAN_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    logger.warning("Mathematician prompt not found. Using default.")
    MATHEMATICIAN_SYSTEM_PROMPT = "You are a clinical risk auditor. Interpret the pre-calculated scores."

def mathematician_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó Matemático Híbrido (Padrão Glass Box).
    """
    logger.info("--- 🧮 NODE: MATHEMATICIAN (Deterministic) ---")
    
    # 1. Recuperação Segura dos Dados (CORREÇÃO AQUI)
    extracted_data = state.get("extracted_data", {})
    
    vitals_data = None
    
    # Estratégia de Fallback para encontrar os dados em diferentes formatos
    if "extracted_vitals" in extracted_data:
        # Formato 'achatado' (Flat) pelo Adapter
        vitals_data = extracted_data["extracted_vitals"]
    elif "vital_signs" in extracted_data:
        # Formato antigo
        vitals_data = extracted_data["vital_signs"]
    elif isinstance(extracted_data, dict) and "clinical" in extracted_data:
        # CORREÇÃO: Formato ScribeSchema puro (dict aninhado)
        # O log mostrou: {'clinical': {'vitals': {...}}}
        vitals_data = extracted_data.get("clinical", {}).get("vitals")
    elif hasattr(extracted_data, "clinical"): 
        # Formato ScribeSchema como Objeto Pydantic
        vitals_data = extracted_data.clinical.vitals.model_dump()
    
    if not vitals_data:
        logger.error(f"No vitals data found in state. State keys: {extracted_data.keys() if isinstance(extracted_data, dict) else 'Not a dict'}")
        return {
            "risk_score_report": "Error: Missing clinical data for calculation.", # Fallback key for supervisor
            "risk_analysis": {
                "overall_risk_assessment": "N/A - Missing Data",
                "capabilities": [],
                "analyzed_scores": []
            }
        }

    # 2. Execução Determinística das Ferramentas (Python Puro)
    logger.info("Executing Python Calculations directly...")
    
    results = {}
    scores_to_run = ["NEWS", "MEWS"]
    
    try:
        # Garante que vitals_data é um dicionário compatível com VitalsSchema
        vitals_obj = VitalsSchema(**vitals_data)
        
        for score_name in scores_to_run:
            try:
                # Chama a função diretamente
                if hasattr(calculate_clinical_score, "invoke"):
                    res = calculate_clinical_score.invoke({"vitals": vitals_obj, "score_name": score_name})
                else:
                    res = calculate_clinical_score(vitals=vitals_obj, score_name=score_name)
                
                results[score_name] = res
            except Exception as e:
                logger.warning(f"Failed to calculate {score_name}: {e}")
                results[score_name] = f"Error: {str(e)}"
                
    except Exception as e:
        logger.error(f"Critical error preparing vitals for calculation: {e}")
        return {
             "risk_score_report": f"Calculation Error: {str(e)}"
        }

    logger.info(f"Calculated Results: {results}")

    # 3. Invocação do Modelo para Interpretação (NLU apenas)
    context_msg = f"""
    [PRE-CALCULATED DATA - DO NOT RECALCULATE]
    Input Vitals: {json.dumps(vitals_data, default=str)}
    
    [CALCULATION RESULTS]
    {json.dumps(results, indent=2)}
    
    [TASK]
    Analyze the calculated scores above.
    1. Fill the 'capabilities' list indicating which scores were successfully calculated.
    2. Fill 'analyzed_scores' for those with valid results.
    3. Synthesize the 'overall_risk_assessment'.
    """
    
    messages = [
        SystemMessage(content=MATHEMATICIAN_SYSTEM_PROMPT),
        HumanMessage(content=context_msg)
    ]
    
    try:
        response = mathematician_model.invoke(messages)
        result_data = response.model_dump()
        
        # Injetamos o resultado bruto do Python para auditoria perfeita
        result_data["calculated_raw"] = results
        
        # Gera um relatório de texto simples para o Supervisor (backward compatibility)
        simple_report = f"NEWS: {results.get('NEWS', 'N/A')}\nMEWS: {results.get('MEWS', 'N/A')}"
        
        return {
            "risk_analysis": result_data,
            "risk_score_report": simple_report # Chave que o Supervisor procura
        }
        
    except Exception as e:
        logger.error(f"Llama 3 Interpretation failed: {e}")
        return {
            "risk_score_report": f"Auto-Calc: {str(results)} (AI Interpretation Failed)"
        }