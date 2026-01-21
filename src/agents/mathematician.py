import os
import logging
import json
import traceback
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.state.agent_state import AgentState
from src.schemas.scribe_schema import VitalsSchema
from src.tools.calculator import calculate_clinical_score
from src.schemas.mathematician_schema import MathematicianSchema

load_dotenv()

# Logger Configuration
logger = logging.getLogger(__name__)

SEED = 42

# 1. LLM Configuration
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
    Executes the Mathematician agent node responsible for calculating and analyzing clinical risk scores.

    This function performs the following steps:
    1.  **Data Extraction**: Retrieves vital signs data from the agent state, handling different data structures (objects or dictionaries).
    2.  **Deterministic Calculation**: Uses the `calculate_clinical_score` tool to compute scores like NEWS (National Early Warning Score) and MEWS (Modified Early Warning Score) based on the extracted vitals. It handles potential calculation errors and missing data imputation warnings.
    3.  **LLM Interpretation**: Invokes a structured LLM (Mathematician Model) to analyze the calculated scores, identify any data gaps (estimated scores), and provide an overall risk assessment.
    4.  **Reporting**: Constructs a comprehensive report including raw calculations, analyzed interpretations, and risk assessments to be updated in the agent state.

    Args:
        state (AgentState): The current state of the agent, expected to contain 'extracted_data' with vital signs.

    Returns:
        Dict[str, Any]: A dictionary containing state updates, specifically:
            - 'extracted_data': The original extracted clinical data (preserved).
            - 'risk_score_report': A simple text summary of the calculated scores.
            - 'risk_analysis': A detailed structured analysis including numeric scores, data quality notes, and risk synthesis.
            - In case of error: Returns an error message in 'risk_score_report'.
    """
    # logger.info("--- Node: Mathematician Agent ---")
    logger.info("\n--- 🧮 NODE: MATHEMATICIAN ---")
    
    try:
        # 1. Retrieve vitals with fallback logic
        data = state.get("extracted_data")
        if not data:
            logger.error("No extracted_data found in state")
            return {"risk_score_report": "Error: Missing Data"}

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

        # 2. Deterministic Execution (Tool Usage by direct Python)
        results = {}
        for score in ["NEWS", "MEWS"]:
            try:
                logger.debug("Calculating scores via tool...")
                results[score] = calculate_clinical_score(vitals_data.model_dump(), score)
            except Exception as e:
                logger.error(f"Calculation Error {score}: {e}")
                results[score] = f"Error calculating {score}: {str(e)}"

        # 3. Model Invocation for Interpretation (NLU)
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
        
        # 4. Risk analysis by LLM
        logger.debug("⏳ Calling LLM for Risk Analysis...")
        response = mathematician_model.invoke(messages)

        result_data = None
        if hasattr(response, "model_dump"):
            result_data = response.model_dump()
        elif hasattr(response, "dict"):
            result_data = response.dict()
            
        if result_data is None:
            logger.error("Data returned in an unsupported format.")
            return  {"risk_score_report": "Data has an unsupported type."}
        
        result_data["calculated_raw"] = results
        simple_report = ""
        for score_name, text in results.items():
            simple_report += f"{score_name}: {text}\n"

        logging.info(f"✅ Mathematician Complete: {simple_report}")
        
        return {
            "extracted_data": data,
            "risk_score_report": simple_report,
            "risk_analysis": result_data 
        }

    except Exception as e:
        logger.error(f"❌ Mathematician Critical Error: {e}")
        traceback.print_exc()
        return {"risk_score_report": f"Critical Error in Mathematician: {str(e)}"}