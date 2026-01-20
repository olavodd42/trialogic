"""
Main execution point for the TriaLogic Agent Graph.

This module defines the agent workflow using LangGraph. It connects the supervisor,
scribe, validator, mathematician, clinical_rag, and synthesizer nodes into a cohesive
state machine.
"""

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from src.state.agent_state import AgentState
from src.agents.supervisor import supervisor_planning, supervisor_router
from src.agents.scribe import ScribeAgent
from src.agents.validator import validator_node, validator_router
from src.agents.mathematician import mathematician_node
from src.agents.clinical_rag import clinical_rag_node
from src.agents.synthesizer import synthesizer_node

def create_workflow():
    # --- LLM SETUP ---
    # Init LLM for Scribe (Dependency Injection)
    llm = ChatOllama(model="llama3.1", temperature=0, seed=42)
    scribe_agent = ScribeAgent(model=llm)

    # --- GRAPH ---
    workflow = StateGraph(AgentState)

    # --- NODES ---
    workflow.add_node("supervisor", supervisor_planning)
    workflow.add_node("scribe", scribe_agent.process)
    workflow.add_node("validator", validator_node)
    workflow.add_node("mathematician", mathematician_node)
    workflow.add_node("clinical_rag", clinical_rag_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # --- EDGES ---
    # workflow.set_entry_point("supervisor")

    # Entry Point
    workflow.set_conditional_entry_point(
        supervisor_router,
        {
            "scribe": "scribe",
            "mathematician": "mathematician",
            "clinical_rag": "clinical_rag",
            "end": END
        }
    )

    # Validation loop
    workflow.add_edge("scribe", "validator")
    workflow.add_conditional_edges(
        "validator",
        validator_router,
        {
            "scribe": "scribe",
            "supervisor": "supervisor"
        }
    )
    
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "scribe": "scribe",
            "mathematician": "mathematician",
            "clinical_rag": "clinical_rag",
            "end": END
        }
    )

    # Return to supervisor
    workflow.add_conditional_edges(
        "mathematician",
        supervisor_router,
        {
            "clinical_rag": "clinical_rag",
            "end": END,
            "auditor": "clinical_rag"} # Mapeie auditor->clinical_rag se tiver string velha
    )

    # Clinical RAG (Retrieval) -> Synthesizer (Audit) -> Supervisor Flow
    workflow.add_edge("clinical_rag", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
