"""
Main execution point for the TriaLogic Agent Graph.

This module defines the agent workflow using LangGraph. It connects the supervisor,
scribe, validator, mathematician, clinical_rag, and synthesizer nodes into a cohesive
state machine.
"""
import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from src.state.agent_state import AgentState
from src.agents.supervisor import supervisor_planning, supervisor_router
from src.agents.scribe import ScribeAgent
from src.agents.validator import validator_node, validator_router
from src.agents.mathematician import MathematicianAgent
from src.agents.clinical_rag import clinical_rag_node
from src.agents.synthesizer import synthesizer_node
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.DEBUG)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY não encontrada no ambiente. Verifique .env / variáveis de ambiente.")

def create_workflow():
    """
    Constructs and compiles the StateGraph for the TriaLogic clinical audit system.

    Builds the graph nodes (Supervisor, Scribe, Validator, Mathematician, RAG, Synthesizer)
    and defines the conditional edges and routing logic that govern the flow of execution.

    Returns:
        CompiledStateGraph: The compiled LangGraph application ready for invocation.
    """
    # --- LLM SETUP ---
    logger.info("Creating workflow...")
    # Init LLM for Scribe (Dependency Injection)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        seed=42
    )
    scribe_agent = ScribeAgent(model=llm)
    mathematician_agent = MathematicianAgent(model=llm)

    # --- GRAPH ---
    workflow = StateGraph(AgentState)

    # --- NODES ---
    import time
    def timed_node(name, func):
        def wrapper(state):
            # Suporte tanto para dict quanto para objeto
            if isinstance(state, dict):
                timings = state.get("_timings", {})
                start = time.time()
                result = func(state)
                elapsed = time.time() - start
                timings[name] = timings.get(name, 0) + elapsed
                state["_timings"] = timings
                return result
            else:
                if not hasattr(state, "_timings"):
                    state._timings = {}
                start = time.time()
                result = func(state)
                elapsed = time.time() - start
                timings = getattr(state, "_timings", {})
                timings[name] = timings.get(name, 0) + elapsed
                state._timings = timings
                return result
        return wrapper

    workflow.add_node("supervisor", supervisor_planning)
    workflow.add_node("scribe", timed_node("scribe", scribe_agent.process))
    workflow.add_node("validator", timed_node("validator", validator_node))
    workflow.add_node("mathematician", timed_node("mathematician", mathematician_agent.process))
    workflow.add_node("clinical_rag", timed_node("clinical_rag", clinical_rag_node))
    workflow.add_node("synthesizer", timed_node("synthesizer", synthesizer_node))

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
            "auditor": "clinical_rag"} 
    )

    # Clinical RAG (Retrieval) -> Synthesizer (Audit) -> Supervisor Flow
    workflow.add_edge("clinical_rag", "synthesizer")
    workflow.add_edge("synthesizer", END)

    compiled = workflow.compile()
    # Adiciona um método para extrair timings do state final
    def get_timings(final_state):
        if isinstance(final_state, dict):
            timings = final_state.get("_timings", {})
        else:
            timings = getattr(final_state, "_timings", {})
        if timings:
            logger.info("Tempo por agente neste caso: " + ", ".join(f"{k}: {v:.2f}s" for k, v in timings.items()))
        return timings
    compiled.get_timings = get_timings
    return compiled