import os
import json
from typing import Dict, Any
from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import logging

from src.state.agent_state import AgentState

load_dotenv()

# Configuração de Logs (Essencial para TCC e Debug)
logger = logging.getLogger(__name__)

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

@lru_cache(maxsize=1)
def get_vectorstore():
    """Singleton connection to ChromaDB."""
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings()
    )

with open(os.path.join(os.getcwd().replace("\\", "/"), "prompts/rag_prompt.md")) as f:
    QUERY_GEN_SYSTEM = f.read()

def clinical_rag_node(state: AgentState) -> Dict[str, Any]:
    print("--- 🔍 NODE: CLINICAL RAG ---")

    # 1. Recovery data from previous nodes
    data = state.get("extracted_data")
    risk = state.get("risk_score_report")

    # 2. Prepare context for query generation
    patient_summary = json.dumps({
        "condition": data.semantics.chief_complaint,
        "vitals": data.clinical.vitals,
        "risk": risk
    })

    # 2. Query Transformation (LLM)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query_msg = [
        SystemMessage(content=QUERY_GEN_SYSTEM),
        HumanMessage(content=f"Patient Data: {patient_summary}")
    ]
    search_query = llm.invoque(query_msg).content

    patient_json = extracted_data.model_dump_json()
    full_patient_context = f"Data: {patient_json}. Risk Assessment: {risk_report}"

    # ----------------------------------------
    # SUB-STEP B: RETRIEVAL (RAG)
    # ----------------------------------------
    vectorstore = get_vectorstore()

    # Filter definition
    retriever_kwargs: Dict[str, Any] = {"k": 3}
    context_category = state.get("context_category")
    if context_category:
            retriever_kwargs["filter"] = {"category": context_category}
    docs = vectorstore.similarity_search(search_query, **retriever_kwargs)

    if docs:
        retrieved_context = "\n\n".join([
            f"[Source: {d.metadata.get('source_type', 'Guideline')}]\n{d.page_content}" 
            for d in docs
        ])
    else:
        retrieved_context = "No specific protocol found in the database."
        print("⚠️ [WARNING]: No documents retrieved.")

    return {
        "search_query": search_query,
        "context_text": retrieved_context
    }