from typing import List, Literal, Dict, Callable
from src.state.agent_state import AgentState
import logging

logger = logging.getLogger(__name__)


def create_supervisor_planning(use_rag: bool = True) -> Callable[[AgentState], Dict[str, List[str]]]:
    """
    Factory: creates a supervisor_planning node parameterized by feature flags.

    Args:
        use_rag (bool): Whether Clinical RAG is enabled in the pipeline.

    Returns:
        A supervisor_planning function compatible with LangGraph.
    """
    def supervisor_planning(state: AgentState) -> Dict[str, List[str]]:
        print("--- 🧠 NODE: SUPERVISOR (PLANNING) ---")
        if use_rag:
            initial_plan = ["scribe", "mathematician", "clinical_rag"]
        else:
            initial_plan = ["scribe", "mathematician"]
        return {"plan": initial_plan}
    return supervisor_planning


def create_supervisor_router(use_rag: bool = True) -> Callable[[AgentState], str]:
    """
    Factory: creates a supervisor_router parameterized by feature flags.

    Args:
        use_rag (bool): Whether Clinical RAG is enabled in the pipeline.

    Returns:
        A supervisor_router function compatible with LangGraph.
    """
    def supervisor_router(state: AgentState) -> str:
        """
        Supervisor Router: Determines the next step in the workflow based on the state.

        Logic:
        1. If no extracted data, route to 'scribe'.
        2. If validation errors exist and attempts < max, retry 'scribe'.
        3. If structured data exists but no risk score, route to 'mathematician'.
        4. If risk scores exist but no contextual grounding, route to 'clinical_rag' (if enabled).
        5. If all steps are complete, terminate execution.
        """
        extracted = state.get("extracted_data")
        risk = state.get("risk_score_report")
        audit = state.get("auditor_report")
        context = state.get("context_text")
        errors = state.get("validation_errors", [])
        attempts = state.get("attempts", 0)

        # 1. Extraction Phase
        if not extracted and attempts < 3:
            return "scribe"

        # 2. Validation Retry Logic
        if errors and attempts < 3:
            logging.warning(f"⚠️ Validation errors detected: {len(errors)}. Retrying Scribe (Attempt {attempts+1}).")
            return "scribe"

        if not extracted and attempts >= 3:
            logging.error("❌ Max attempts reached for extraction. Giving up.")
            return "end"

        # 3. Calculation Phase
        if extracted and not risk:
            return "mathematician"

        # 4. RAG Phase (only if enabled), otherwise go to synthesizer
        if risk and not audit:
            if use_rag:
                return "clinical_rag"
            else:
                return "synthesizer"

        return "end"
    return supervisor_router


# Backward-compatible defaults (used when importing directly)
supervisor_planning = create_supervisor_planning(use_rag=True)
supervisor_router = create_supervisor_router(use_rag=True)