import chromadb
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from typing import Dict, Any
from src.agents.agent_state import AgentState
from src.retriever_data.load_and_preprocess import load_pdf

def query_transformation(state: AgentState) -> AgentState:
    patient_schema = state["extracted_data"].model_dump_json()
    score_report = state["risk_score_report"]
    system_prompt = SystemMessage(content="""
        You are a Clinical Knowledge Retrieval Specialist utilizing a Vector Database (RAG) to audit medical cases.
        Your sole responsibility is to translate a structured 'Patient State' (JSON) into a precise, semantic search query suitable for retrieving clinical protocols (e.g., NICE, Sepsis-3, ACLS).

        ### Instructions:
        1. ANALYZE the Patient State: Focus heavily on abnormal vital signs, high risk scores (NEWS/MEWS), and chief complaints.
        2. ABSTRACTION: Do not just repeat numbers. Translate numbers into clinical terms (e.g., HR 130 -> 'Tachycardia', Temp 39 -> 'Fever/Pyrexia').
        3. TARGET: Formulate a query that targets the *management guidelines* or *standard of care* for the identified condition.
        4. PRIVACY: REMOVE any patient identifiers (Names, IDs). The query must be anonymous.

        ### Output Format:
        Return ONLY the search query string. No explanations, no markdown, no quotes.

        ### Examples:

        Input:
        {
        "chief_complaint": "chest pain radiating to left arm",
        "vitals": {"hr": 110, "bp": "160/95", "spo2": 98},
        "risk_score": "NEWS: 4"
        }
        Output:
        Acute Coronary Syndrome chest pain management protocol

        Input:
        {
        "chief_complaint": "shortness of breath",
        "vitals": {"hr": 125, "bp": "85/50", "temp": 39.2},
        "risk_score": "NEWS: 11 (High Risk)"
        }
        Output:
        Sepsis-3 guidelines hypotension tachycardia fever management

        Input:
        {
        "chief_complaint": "fall, hit head",
        "vitals": {"gcs": 13, "bp": "130/80"},
        "risk_score": "Trauma Score: Moderate"
        }
        Output:
        Traumatic Brain Injury TBI adult triage guidelines head trauma
    """)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    user_prompt = HumanMessage(content=f"Collected data about the patient {patient_schema} and score report \"{score_report}\"")
    search_query = str(llm.invoke([system_prompt, user_prompt]).content)
    state["search_query"] = search_query
    print(f"🔍 Generated Query: {search_query}")
    
    return state

def retriever(state: AgentState, vectorstore: chromadb.Collection) -> AgentState:
    query = state["search_query"]
    category_filter = state.get("context_category", None)
    search_kwargs: Dict[str, Any] = {"k": 3}

    if category_filter:
        search_kwargs["filter"] = {"category": category_filter}

    docs = vectorstore.similarity_search(query, **search_kwargs)
    context_text = "\n\n".join([d.page_content for d in docs])
    state["context_text"] = context_text

    return state

def synthesis_and_audit(state: AgentState) -> AgentState:
    text = state.get("context_text", "")
    system_prompt = SystemMessage(content="")
    return state

def auditor_node(state: AgentState) -> AgentState:
    return state