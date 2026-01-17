import os
import logging
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
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
# Removemos a complexidade de tools para destravar o Llama 3.
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key=SecretStr("lm-studio"),
    model="meta-llama-3.1-8b-instruct",
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
    
    Mudança de Paradigma:
    1. Python executa o cálculo determinístico (NEWS e MEWS).
    2. Llama 3 recebe os resultados prontos e apenas gera a interpretação clínica.
    Isso elimina o congelamento por falha de Function Calling em modelos menores.
    """
    logger.info("--- 🧮 NODE: MATHEMATICIAN (Deterministic) ---")
    
    # 1. Recuperação Segura dos Dados
    # Tenta pegar os vitais limpos do Adapter (preferencial) ou brutos
    extracted_data = state.get("extracted_data", {})
    
    vitals_data = None
    
    # Estratégia de Fallback para encontrar os dados
    if "extracted_vitals" in extracted_data:
        vitals_data = extracted_data["extracted_vitals"]
    elif "vital_signs" in extracted_data:
        vitals_data = extracted_data["vital_signs"]
    elif hasattr(extracted_data, "clinical"): # Caso venha como objeto Pydantic
        vitals_data = extracted_data.clinical.vitals.model_dump()
    
    if not vitals_data:
        logger.error("No vitals data found in state.")
        return {
            "risk_analysis": {
                "risk_score": "N/A",
                "reasoning": "Missing clinical data for calculation."
            }
        }

    # 2. Execução Determinística das Ferramentas (Python Puro)
    # Não pedimos para a IA calcular. Nós calculamos.
    logger.info("Executing Python Calculations directly...")
    
    results = {}
    scores_to_run = ["NEWS", "MEWS"]
    
    try:
        # Reconstrói o objeto VitalsSchema necessário para a ferramenta
        # O Adapter do Scribe garante que vitals_data seja um dict plano compatível
        vitals_obj = VitalsSchema(**vitals_data)
        
        for score_name in scores_to_run:
            try:
                # Chama a função diretamente (bypassando a decisão da IA)
                # Se 'calculate_clinical_score' for uma @tool, usamos .invoke ou chamamos direto
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
        return {"risk_analysis": {"risk_score": "Error", "reasoning": str(e)}}

    logger.info(f"Calculated Results: {results}")

    # 3. Invocação do Modelo para Interpretação (NLU apenas)
    # O Llama 3 agora só precisa ler o texto e formatar o JSON. Muito mais fácil.
    
    context_msg = f"""
    [PRE-CALCULATED DATA - DO NOT RECALCULATE]
    Input Vitals: {json.dumps(vitals_data, default=str)}
    
    [CALCULATION RESULTS]
    {json.dumps(results, indent=2)}
    
    [TASK]
    Analyze the calculated scores above.
    1. Summarize the risk level.
    2. Provide clinical reasoning based strictly on these numbers.
    3. Output in the required JSON format.
    """
    
    messages = [
        SystemMessage(content=MATHEMATICIAN_SYSTEM_PROMPT),
        HumanMessage(content=context_msg)
    ]
    
    try:
        # A chamada agora é rápida e não trava
        response = mathematician_model.invoke(messages)
        result_data = response.model_dump()
        
        # Injetamos o resultado bruto do Python para auditoria perfeita
        result_data["calculated_raw"] = results
        
        return {
            "risk_analysis": result_data
        }
        
    except Exception as e:
        logger.error(f"Llama 3 Interpretation failed: {e}")
        # Fallback se até a interpretação falhar (mas o cálculo já temos!)
        return {
            "risk_analysis": {
                "risk_score": str(results),
                "reasoning": "Automated Calculation (AI Interpretation Failed)",
                "calculated_raw": results
            }
        }