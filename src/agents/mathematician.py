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
from src.tools.calculator import calculate_clinical_score 
from src.schemas.mathematician_schema import MathematicianSchema

load_dotenv()

# Logger Configuration
logger = logging.getLogger(__name__)

SEED = 42

# 1. Configuração do LLM
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.1",
    temperature=0,
    seed=SEED
)

# Structured Output
mathematician_model = llm.with_structured_output(MathematicianSchema)

# Load Prompt
prompt_path = os.path.join(os.getcwd(), "prompts", "mathematician_prompt.md")
try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        MATHEMATICIAN_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    logger.warning("Mathematician prompt not found. Using default.")
    MATHEMATICIAN_SYSTEM_PROMPT = "You are a clinical mathematician agent."

def mathematician_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent responsible for deterministic calculation of clinical scores.
    Refactored to handle nested state structures (clinical -> vitals).
    """
    logger.info("--- Node: Mathematician Agent ---")
    
    # 1. Recupera Vitais do Estado com Lógica de Fallback (Tech Lead Fix)
    vitals_data = state.get("vitals")
    
    # Se não achar na raiz, procura dentro de 'clinical' (onde o Scribe geralmente coloca)
    if not vitals_data:
        logger.info("Vitals not found at root. Checking 'clinical' nested key...")
        clinical_data = state.get("clinical", {})
        if clinical_data:
            vitals_data = clinical_data.get("vitals")

    # Se ainda assim for None ou vazio
    if not vitals_data:
        logger.error("No vitals found in state (checked root and 'clinical.vitals')")
        return {
            "risk_analysis": {"error": "No vitals extracted"},
            "risk_score": "No vitals available for calculation"
        }

    # Converter dicionário para Schema Pydantic para validação/uso
    try:
        vitals_schema = VitalsSchema(**vitals_data)
    except Exception as e:
        logger.error(f"Vitals validation failed: {e}")
        return {
            "risk_analysis": {"error": str(e)},
            "risk_score": "Vitals data validation error"
        }

    # 2. Execução Determinística (Tool Usage via Python direto)
    results = {}
    for score in ["NEWS", "MEWS"]:
        # A ferramenta agora retorna string formatada com Warnings se houver imputação
        results[score] = calculate_clinical_score(vitals_schema, score)

    # 3. Invocação do Modelo para Interpretação (NLU)
    context_msg = f"""
    [PRE-CALCULATED SCORES]
    Analyze the following calculation outputs carefully. Note any [ESTIMATED] tags.
    
    Input Vitals: {json.dumps(vitals_data, default=str)}
    
    Calculation Output:
    {json.dumps(results, indent=2)}
    
    [TASK]
    Analyze the calculated scores above.
    1. If the score status is [ESTIMATED], mention explicitly which data was missing in your analysis.
    2. Fill 'analyzed_scores' with the numeric values.
    3. Synthesize the 'overall_risk_assessment'.
    """
    
    messages = [
        SystemMessage(content=MATHEMATICIAN_SYSTEM_PROMPT),
        HumanMessage(content=context_msg)
    ]
    
    try:
        response = mathematician_model.invoke(messages)
        result_data = response.model_dump()
        
        # Injetamos o resultado bruto para auditoria
        result_data["calculated_raw"] = results
        
        # Relatório simples para o Supervisor/Auditor ler
        simple_report = ""
        for score_name, text in results.items():
            # Tenta extrair o valor numérico de forma segura
            try:
                score_val = text.split('\n')[1] if "SCORE TOTAL" in text else "N/A"
            except IndexError:
                score_val = "Error Parsing"
                
            if "[ESTIMATED]" in text:
                simple_report += f"{score_name}: {score_val} (ESTIMATED - Missing Data)\n"
            else:
                simple_report += f"{score_name}: {score_val}\n"

        return {
            "risk_analysis": result_data,
            "risk_score": simple_report 
        }

    except Exception as e:
        logger.error(f"Mathematician LLM error: {e}")
        return {
            "risk_analysis": {"error": str(e)},
            "risk_score": "Error in risk analysis interpretation"
        }