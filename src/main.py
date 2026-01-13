"""
Main execution point for the TriaLogic Agent Graph.

This module defines the agent workflow using LangGraph. It connects the supervisor,
scribe, validator, mathematician, clinical_rag, and synthesizer nodes into a cohesive
state machine.
"""

from langgraph.graph import StateGraph, END

from src.state.agent_state import AgentState
from src.agents.supervisor import supervisor_planning, supervisor_router
from src.agents.scribe import scribe_node
from src.agents.validator import validator_node, validator_router
from src.agents.mathematician import mathematician_node
from src.agents.clinical_rag import clinical_rag_node
from src.agents.synthesizer import synthesizer_node

# Initialize the State Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_planning)
workflow.add_node("scribe", scribe_node)
workflow.add_node("validator", validator_node)
workflow.add_node("mathematician", mathematician_node)
workflow.add_node("clinical_rag", clinical_rag_node)
workflow.add_node("synthesizer", synthesizer_node)

# Set Entry Point
workflow.set_entry_point("supervisor")

# Supervisor Routing
# The supervisor decides which agent to call next based on the state.
workflow.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "scribe": "scribe",
        "mathematician": "mathematician",
        "auditor": "clinical_rag", # Route 'auditor' task to clinical_rag node (start of audit pipeline)
        "end": END
    }
)

# Scribe -> Validator Flow
workflow.add_edge("scribe", "validator")

# Validator Routing
# If validation fails, retry scribe. Else, return to supervisor to plan next step.
workflow.add_conditional_edges(
    "validator",
    validator_router,
    {
        "scribe": "scribe",
        "supervisor": "supervisor"
    }
)

# Mathematician -> Supervisor Flow
workflow.add_edge("mathematician", "supervisor")

# Clinical RAG (Retrieval) -> Synthesizer (Audit) -> Supervisor Flow
workflow.add_edge("clinical_rag", "synthesizer")
workflow.add_edge("synthesizer", "supervisor")

# Compile the graph
app = workflow.compile()
