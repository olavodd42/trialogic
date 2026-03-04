"""Clinical RAG (Retrieval-Augmented Generation) agent node.

This module handles querying a ChromaDB vector store of clinical guidelines
and protocols to provide evidence-based context for the Synthesizer agent.
"""

import os
import re
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from src.state.agent_state import AgentState
from src.utils.vectorstore import get_vectorstore
from src.utils.run_with_timeout import run_with_timeout

load_dotenv()

# Logger configuration
logger = logging.getLogger(__name__)

# --- Directory configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")

path = os.path.join(project_root, "prompts/rag_prompt.md")
try:
    # Use absolute paths based on the project root for reliability
    full_path = os.path.join(os.getcwd(), path)
    with open(full_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
except FileNotFoundError:
    logger.error("CRITICAL: Prompt file not found at %s", path)
    raise RuntimeError(f"RAG system prompt missing: {path}") 

GUIDELINE_KEYWORDS = [
    "recommend", "should", "must", "indicated",
    "management", "treatment", "protocol",
    "guideline", "assessment", "ct scan"
]


def _extract_clean_query(raw_response: str) -> str:
    """
    Post-processes the LLM response to extract only the search query.
    LLaMA 3.1 often outputs full reasoning before the actual query.
    Strategy:
      1. Look for quoted strings — pick the first clinically-relevant one.
      2. If multiple short quotes, join the first 2.
      3. If no quotes, take the last non-empty short line.
      4. Fallback to first 200 chars.
    """
    MAX_QUERY_LEN = 200

    # 1. Find all quoted strings (10..500 chars — LLaMA sometimes generates long quotes)
    quoted = re.findall(r'"([^"]{10,500})"', raw_response)
    if quoted:
        # Filter out meta-phrases that are not real queries
        real_queries = [
            q for q in quoted
            if not q.lower().startswith(
                ("based on", "here is", "note that", "this query", "considering")
            )
        ]
        if not real_queries:
            real_queries = quoted
        
        if len(real_queries) == 1:
            return real_queries[0][:MAX_QUERY_LEN]
        
        # Multiple queries: pick the first one (most relevant to primary condition)
        # If it's short (<60 chars), combine with the second for richer context
        first = real_queries[0]
        if len(first) < 60 and len(real_queries) > 1:
            combined = f"{first} {real_queries[1]}"
            return combined[:MAX_QUERY_LEN]
        return first[:MAX_QUERY_LEN]

    # 2. Look for lines that look like a query (no markdown headers, short-ish)
    lines = [l.strip() for l in raw_response.strip().split("\n") if l.strip()]
    candidate_lines = [
        l for l in lines
        if 15 < len(l) < 300
        and not l.startswith(("**", "#", "-", "*"))
        and not l.lower().startswith(
            ("note", "based on", "here is", "this query", "considering", "since", "we will")
        )
    ]
    if candidate_lines:
        return candidate_lines[-1][:MAX_QUERY_LEN]

    # 3. Fallback
    return raw_response[:MAX_QUERY_LEN]


def is_actionable_guideline(docs: List[Document]) -> bool:
    """Check whether retrieved documents contain actionable clinical guidelines.

    A document is considered actionable if it contains at least two
    guideline-related keywords (e.g. 'recommend', 'protocol', 'treatment').

    Args:
        docs: List of retrieved LangChain Document objects.

    Returns:
        True if at least one document meets the actionability threshold.
    """
    for d in docs:
        text = d.page_content.lower()
        hits = sum(1 for k in GUIDELINE_KEYWORDS if k in text)
        if hits >= 2:
            return True
    return False

def clinical_rag_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Clinical RAG (Retrieval-Augmented Generation) node to fetch relevant clinical guidelines.

    This node bridges the gap between raw patient data and established medical knowledge by:
    1.  **Context Extraction**: Pulling patient vitals, risk scores, and a snippet of the clinical note from the agent state.
    2.  **Query Generation**: Using an LLM to formulate an optimized search query based on the patient's specific condition and the RAG system prompt.
    3.  **Vector Search**: Querying the local ChromaDB vector store to find the top-k most relevant clinical guidelines or protocols.
    4.  **Context formatting**: Formatting the retrieved documents into a concise string for downstream consumption by the Synthesizer agent.

    Args:
        state (AgentState): The current state of the agent, expected to contain 'input' (raw text), 'vitals', and 'risk_score_report'.

    Returns:
        Dict[str, Any]: A dictionary containing state updates:
            - 'search_query': The actual query generated by the LLM.
            - 'rag_context': A concatenated string of retrieved guidelines (or a fallback message if retrieval fails).
    """
    logger.info("--- NODE: CLINICAL RAG ---")
    
    # 1. Input extraction with expanded context
    raw_text = state['input'].raw_text[:2000] 
    vitals = state.get("vitals", {})
    risk_score = state.get("risk_score_report", "Risk Score not calculated")
    
    # 2. LLM Configuration (Query Generator)
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
        seed=42,
        num_ctx=8192,
    )

    # 3. Prompt (the RAG system constitution)
    user_message = f"""
    PATIENT VITALS: {vitals}
    RISK SCORE: {risk_score}
    
    CLINICAL NOTE SNIPPET (READ CAREFULLY):
    "{raw_text}..."
    
    Generate the optimized search query:
    """

    search_query = "Standard clinical guidelines" # Safe Fallback
    retrieved_context = "Guidelines unavailable due to error."

    try:
        # 4. Query Generation
        logger.debug("Generating Search Query via LLM...")
        response = run_with_timeout(
            llm.invoke,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)],
            timeout=120, retries=2
        )
        raw_query = response.content.strip()
        search_query = _extract_clean_query(raw_query)
        logger.info("RAG Query (clean): '%s'", search_query)
        logger.debug("RAG Query (raw, %d chars): '%s...'", len(raw_query), raw_query[:120])

        # 5. Similarity search
        logger.debug("Accessing VectorStore...")
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(search_query, k=3)
        if docs and is_actionable_guideline(docs):
            rag_context_used = True
        else:
            rag_context_used = False
        
        if rag_context_used:
            formatted_docs = []
            for d in docs:
                source = d.metadata.get('source', 'Guideline')
                content = d.page_content[:600].replace("\n", " ") 
                formatted_docs.append(f"SOURCE [{source}]: {content}...")
            
            retrieved_context = "\n\n".join(formatted_docs)
            logger.info("Retrieval successful (%d docs found).", len(docs))
        else:
            logger.warning("Retrieval returned 0 actionable documents.")
            retrieved_context = "No specific guidelines found for this query."

    except Exception as e:
        rag_context_used = False
        logger.error("RAG system failed: %s", e)
        retrieved_context = "CRITICAL: RAG System Offline. Proceed with standard clinical judgment."

    return {
        # "extracted_vitals": vitals,
        "search_query": search_query,
        "rag_context": retrieved_context,
        "rag_context_used": rag_context_used
    }