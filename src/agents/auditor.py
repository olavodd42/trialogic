import json
from typing import Dict, Any, cast
from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.agents.agent_state import AgentState
from src.schemas.auditor_schema import AuditorEvaluation

# --- SETUP (Singleton Pattern to the Database) ---
@lru_cache(maxsize=1)
def get_vectorstore():
    """Carrega o banco vetorial apenas uma vez em memória."""
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=OpenAIEmbeddings()
    )

# --- PROMPTS ---
QUERY_GEN_SYSTEM = """
You are a Clinical Knowledge Retrieval Specialist.
Translate the patient state into a semantic search query for medical guidelines.
Focus on abnormal vitals and risk scores.
Output ONLY the query string.
"""

AUDITOR_SYSTEM = """
You are a Senior Clinical Auditor AI. 
Your job is to compare the Patient State against the provided Official Medical Protocols.

CONTEXT (Official Guidelines):
{context}

PATIENT STATE:
{patient_state}

INSTRUCTIONS:
1. Verify if the patient's vitals and scores align with the protocol's severity criteria.
2. Quote the specific line from the context that supports your finding.
3. Determine compliance (Compliant/Non-Compliant).
4. Suggest the next step based on the text.
"""

# --- NODE LOGIC ---
def auditor_node(state: AgentState) -> dict | AgentState:
    print("--- ⚖️ NODE: AUDITOR ---")

    # 1. Recovery data from previous nodes
    extracted_data = state.get("extracted_data")
    risk_report = state.get("risk_score_report")

    if not extracted_data:
        return {"auditor_report": "Error: No data to audit."}
    
    # 2. Prepare serialized input
    patient_json = extracted_data.model_dump_json()
    full_patient_context = f"Data: {patient_json}. Risk Assessment: {risk_report}"

    # ----------------------------------------
    # SUB-STEP A: QUERY TRANSFORMATION
    # ----------------------------------------
    llm_query = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query_msg = [
        SystemMessage(content=QUERY_GEN_SYSTEM),
        HumanMessage(content=full_patient_context)
    ]
    search_query = llm_query.invoke(query_msg).content
    print(f"🔍 Generated Query: {search_query}")

    # ----------------------------------------
    # SUB-STEP B: RETRIEVAL (RAG)
    # ----------------------------------------
    vectorstore = get_vectorstore()

    # Filter definition
    retriever_kwargs: Dict[str, Any] = {"k": 3}
    if state.get("context_category"):
         retriever_kwargs["filter"] = {"context_category": state["context_category"]}

    docs = vectorstore.similarity_search(search_query, **retriever_kwargs)

    retrieved_context = "\n\n".join([f"[Source: {d.metadata.get('source_type', 'Unknown')}]\n{d.page_content}" for d in docs])

    if not docs:
        retrieved_context = "No specific protocol found in the database."
        print("⚠️ Warning: No documents retrieved.")

    # ----------------------------------------
    # SUB-STEP C: SYNTHESIS & STRUCTURED AUDIT
    # ----------------------------------------
    llm_auditor = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    structured_llm = llm_auditor.with_structured_output(AuditorEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", AUDITOR_SYSTEM),
        ("human", "Audit this case.")
    ])
    
    chain = prompt | structured_llm

    try:
        evaluation = cast(AuditorEvaluation, chain.invoke({
            "context": retrieved_context,
            "patient_state": full_patient_context
        }))
        
        print(f"📝 Veredito: {evaluation.compliance}")

        return {
            "search_query": search_query,
            "context_text": retrieved_context,
            "auditor_report": evaluation.model_dump() # Salva como dict
        }
    
    except Exception as e:
        print(f"❌ Error in synthesis: {e}")
        return {"auditor_report": {"error": str(e)}}