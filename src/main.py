"""
Main execution point for the TriaLogic Agent Graph.

This module defines the agent workflow using LangGraph. It connects the supervisor,
scribe, validator, mathematician, clinical_rag, and synthesizer nodes into a cohesive
state machine.

Feature flags:
    use_validator (bool): Enable/disable the Validator node (physiological plausibility checks).
    use_rag (bool): Enable/disable Clinical RAG retrieval node. When disabled, the
        pipeline skips clinical_rag and goes directly from Mathematician to Synthesizer.
    use_probabilistic (bool): If True, Mathematician uses LLM-based scoring; if False, deterministic calculator.
"""
import os
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from src.state.agent_state import AgentState
from src.agents.supervisor import create_supervisor_planning, create_supervisor_router
from src.agents.scribe import ScribeAgent
from src.agents.validator import validator_node, validator_router
from src.agents.mathematician import MathematicianAgent
from src.agents.clinical_rag import clinical_rag_node
from src.agents.synthesizer import synthesizer_node
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)
logging.getLogger("langchain").setLevel(logging.DEBUG)


def create_workflow(
    use_validator: bool = True,
    use_rag: bool = True,
    use_probabilistic: bool = False,
):
    """
    Constructs and compiles the StateGraph for the TriaLogic clinical audit system.

    Args:
        use_validator (bool): If True, Scribe output is validated before proceeding.
            When False, the Validator node is skipped (scribe -> supervisor directly).
        use_rag (bool): If True, Clinical RAG retrieval is included before the Synthesizer.
            When False, the Mathematician output goes directly to the Synthesizer (no RAG).
        use_probabilistic (bool): If True, the Mathematician uses LLM-based probabilistic
            scoring. If False, uses the deterministic calculator.

    Returns:
        CompiledStateGraph: The compiled LangGraph application ready for invocation.
    """
    # --- LLM SETUP ---
    logger.info(
        "Creating workflow (validator=%s, rag=%s, probabilistic=%s)",
        use_validator, use_rag, use_probabilistic,
    )
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
        seed=42,
        num_ctx=8192,
    )
    scribe_agent = ScribeAgent(model=llm)
    mathematician_agent = MathematicianAgent(model=llm, use_probabilistic=use_probabilistic)

    # Build parameterized supervisor functions
    supervisor_planning = create_supervisor_planning(use_rag=use_rag)
    supervisor_router = create_supervisor_router(use_rag=use_rag)

    # --- GRAPH ---
    workflow = StateGraph(AgentState)

    # --- NODES ---
    import time

    def timed_node(name, func):
        def wrapper(state):
            start = time.time()
            result = func(state)
            elapsed = time.time() - start
            if isinstance(result, dict):
                timings = dict(state.get("_timings", {})) if isinstance(state, dict) else {}
                timings[name] = timings.get(name, 0) + elapsed
                result["_timings"] = timings
            return result
        return wrapper

    workflow.add_node("supervisor", supervisor_planning)
    workflow.add_node("scribe", timed_node("scribe", scribe_agent.process))
    workflow.add_node("mathematician", timed_node("mathematician", mathematician_agent.process))
    workflow.add_node("synthesizer", timed_node("synthesizer", synthesizer_node))

    if use_validator:
        workflow.add_node("validator", timed_node("validator", validator_node))

    if use_rag:
        workflow.add_node("clinical_rag", timed_node("clinical_rag", clinical_rag_node))

    # --- ENTRY POINT ---
    entry_map = {
        "scribe": "scribe",
        "mathematician": "mathematician",
        "synthesizer": "synthesizer",
        "end": END,
    }
    if use_rag:
        entry_map["clinical_rag"] = "clinical_rag"

    workflow.set_conditional_entry_point(supervisor_router, entry_map)

    # --- EDGES ---

    # Scribe output: validator loop or direct to supervisor
    if use_validator:
        workflow.add_edge("scribe", "validator")
        workflow.add_conditional_edges(
            "validator",
            validator_router,
            {
                "scribe": "scribe",
                "supervisor": "supervisor",
            },
        )
    else:
        workflow.add_edge("scribe", "supervisor")

    # Supervisor routing
    supervisor_map = {
        "scribe": "scribe",
        "mathematician": "mathematician",
        "synthesizer": "synthesizer",
        "end": END,
    }
    if use_rag:
        supervisor_map["clinical_rag"] = "clinical_rag"

    workflow.add_conditional_edges("supervisor", supervisor_router, supervisor_map)

    # Mathematician exit
    if use_rag:
        workflow.add_conditional_edges(
            "mathematician",
            supervisor_router,
            {
                "clinical_rag": "clinical_rag",
                "end": END,
                "auditor": "clinical_rag",
            },
        )
        # Clinical RAG -> Synthesizer -> END
        workflow.add_edge("clinical_rag", "synthesizer")
    else:
        # Without RAG, mathematician goes directly to synthesizer
        workflow.add_conditional_edges(
            "mathematician",
            supervisor_router,
            {
                "synthesizer": "synthesizer",
                "end": END,
            },
        )

    # Synthesizer always terminates the graph
    workflow.add_edge("synthesizer", END)

    compiled = workflow.compile()

    # Helper to extract per-node timings from final state
    def get_timings(final_state):
        if isinstance(final_state, dict):
            timings = final_state.get("_timings", {})
        else:
            timings = getattr(final_state, "_timings", {})
        if timings:
            logger.info(
                "Tempo por agente neste caso: "
                + ", ".join(f"{k}: {v:.2f}s" for k, v in timings.items())
            )
        return timings

    compiled.get_timings = get_timings
    return compiled