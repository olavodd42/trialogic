from langgraph.graph import StateGraph, END

from src.agents.scribe import AgentState, scribe_node, validator
from src.schemas.input_schema import InputSchema
from src.schemas.scribe_output_schema import ScribeOutputSchema

workflow = StateGraph(AgentState)
workflow.add_node("scribe", scribe_node)

workflow.set_entry_point("scribe")