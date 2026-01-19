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


import traceback

def mathematician_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent responsible for deterministic calculation of clinical scores.
    Refactored to handle nested state structures (clinical -> vitals).
    """
    logger.info("--- Node: Mathematician Agent ---")
    print("\n--- 🧮 NODE: MATHEMATICIAN ---")
    
    try:
        # 1. Recupera Vitais do Estado com Lógica de Fallback (Tech Lead Fix)
        data = state.get("extracted_data")
        if not data:
            raise ValueError("No extracted_data found in state")

        # Handle specific schema access
        if hasattr(data, "vitals"):
            vitals_data = data.vitals
        elif isinstance(data, dict):
            vitals_data = data.get("vitals")
        else:
            vitals_data = None
            

        if not vitals_data:
            logger.error("data has no attribute/key 'vitals'.")
            return {"risk_score_report": "Error: Missing Vitals"}

        # 2. Execução Determinística (Tool Usage via Python direto)
        results = {}
        for score in ["NEWS", "MEWS"]:
            # A ferramenta agora retorna string formatada com Warnings se houver imputação
            try:
                results[score] = calculate_clinical_score(vitals_data, score)
            except Exception as e:
                logger.error(f"Calculation Error {score}: {e}")
                results[score] = f"Error calculating {score}: {str(e)}"

        # 3. Invocação do Modelo para Interpretação (NLU)
        # Safe dump for JSON
        vitals_json = "{}"
        if hasattr(vitals_data, "model_dump"):
            vitals_json = json.dumps(vitals_data.model_dump(), default=str)
        elif hasattr(vitals_data, "dict"):
            vitals_json = json.dumps(vitals_data.dict(), default=str)
        else:
            vitals_json = json.dumps(vitals_data, default=str)

        context_msg = f"""
        [PRE-CALCULATED SCORES]
        Analyze the following calculation outputs carefully. Note any [ESTIMATED] tags.
        
        Input Vitals: {vitals_json}
        
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
        
        print("⏳ Calling LLM for Risk Analysis...")
        response = mathematician_model.invoke(messages)
        result_data = response.model_dump()
        # Injetamos o resultado bruto para auditoria
        result_data["calculated_raw"] = results
        
        # Relatório simples para o Supervisor/Auditor ler
        simple_report = ""
        for score_name, text in results.items():
            simple_report += f"{score_name}: {text}\n"

        print(f"✅ Mathematician Complete.")
        
        return {
            "extracted_data": data,
            "risk_score_report": simple_report,
            "risk_analysis": result_data 
        }

    except Exception as e:
        print(f"❌ Mathematician Critical Error: {e}")
        traceback.print_exc()
        return {"risk_score_report": f"Critical Error in Mathematician: {str(e)}"}