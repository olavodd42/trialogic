import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
from dotenv import load_dotenv
from typing import Dict, Any

from src.state.agent_state import AgentState
from src.tools.calculator import calculate_clinical_score
from src.schemas.scribe_schema import VitalsSchema

load_dotenv()

# 1. Configuração Correta do LLM com Tools
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key=SecretStr("lm-studio"),
    model="gpt-4o-mini",
    temperature=0
)
# Tech Lead Tip: Save the object with bind in a new variable!
llm_with_tools = llm.bind_tools([calculate_clinical_score])

with open(os.path.join(os.getcwd(), "prompts/mathematician_prompt.md")) as f:
    system_msg = f.read()

def mathematician_node(state: AgentState) -> Dict[str, Any]:
    """
    The Mathematician Node:
    1. Analyzes the extracted vitals.
    2. Decides which score is pertinent (or both).
    3. Invokes the deterministic tool (Python).
    4. Updates the state with the auditable result.

    Args:
        state (AgentState): Current state of the workflow.

    Returns:
        dict: Partial state update containing the risk score report.
    """
    
    # Recover data from State (Scribe Output)
    data = state.get("extracted_data")
    if not data or not data.clinical or not data.clinical.vitals:
        return {
            "risk_score_report": "No valid clinical data to calculate scores."
        }
    if hasattr(data, "clinical"):
        if hasattr(data.clinical, "vitals"):
          vitals = data.clinical.vitals
        else:
            raise AttributeError(f"{type(data.clinical)} has no attribute 'clinical'.")
    else:
        raise AttributeError(f"{type(data)} has no attribute 'vitals'.")
    
    # 2. Prompt Focused on ACTION (Tool Call), not Mental Calculation
    
    # Serialize Pydantic object to JSON for the LLM
    vitals_json = vitals.model_dump_json()
    
    user_msg = HumanMessage(content=f"Patient Vital Signs: {vitals_json}")
    
    # 3. Model Invocation
    response = llm_with_tools.invoke([system_msg, user_msg])
    print(system_msg)
    print(user_msg)
    print(f"[ASSISTANT]: {response}")
    
    # 4. Tool Execution (Manual Execution for full control)
    tool_outputs = []
    
    if response.tool_calls:
        print(f"DEBUG: Agent decided to call {len(response.tool_calls)} tools.")
        
        for tool_call in response.tool_calls:
            # LangChain has already parsed the arguments for us
            tool_args = tool_call["args"]
            
            # Reconstruct VitalsSchema object to pass to Python function
            try:
                vitals_obj = VitalsSchema(**tool_args['vitals'])
                score_name = tool_args['score_name']
                
                # Execute deterministic Python function
                result_str = calculate_clinical_score.invoke({"vitals": vitals_obj, "score_name": score_name})
                print(f"RESULT: {result_str}")
                
                tool_outputs.append(result_str)
                
            except Exception as e:
                tool_outputs.append(f"Error executing calculation: {str(e)}")
    else:
        print("DEBUG: Agent did not call any tools.")
        tool_outputs.append("No score calculated (insufficient data or agent decision).")

    # 5. State Update
    # Unify results into a consolidated string for the next agent
    final_score_report = "\n".join(tool_outputs)
    
    # Return only the state delta to update
    return {
        "risk_score_report": final_score_report
    }