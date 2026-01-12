from langgraph.graph import StateGraph, END

from src.state.agent_state import AgentState
from src.agents.supervisor import supervisor_planning, supervisor_router
from src.agents.scribe import scribe_node, validator
from src.agents.mathematician import mathematician_node
from src.agents.auditor import auditor_node
from src.schemas.input_schema import InputSchema
from src.schemas.scribe_schema import ScribeOutputSchema

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_planning)
workflow.add_node("scribe", scribe_node)
workflow.add_node("mathematician", mathematician_node)
workflow.add_node("auditor", auditor_node)

workflow.set_entry_point("supervisor")
workflow.set_conditional_entry_point(
        supervisor_router,
        {
            "scribe": "scribe",
            "mathematician": "mathematician",
            "auditor": "auditor",
            "END": END
        }
)

workflow.add_conditional_edges(
    "scribe",
    supervisor_router,
    {
        "mathematician": "mathematician",
        "auditor": "auditor",
        "END": END
    }
)

workflow.add_conditional_edges(
    "mathematician",
    supervisor_router,
    {
        "auditor": "auditor",
        "END": END
    }
)

workflow.add_edge("auditor", END)