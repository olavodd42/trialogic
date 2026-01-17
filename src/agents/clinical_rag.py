"""
Clinical RAG Agent Module - Smart Fallback Edition (TCC Safe).

Features:
- Circuit Breaker for Vector Store
- Prompt Token Optimization
- Smart Python Fallback: Generates queries deterministically if LLM times out.
"""

import os
import logging
from typing import Dict, Any, List
from functools import lru_cache
from pydantic import SecretStr

# Imports seguros do LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.state.agent_state import AgentState

load_dotenv()

# Logger Configuration
logger = logging.getLogger(__name__)

SEED = 42

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")

# --- PROMPT LOADING ---
prompt_path = os.path.join(os.getcwd(), "prompts", "rag_prompt.md")
try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        RAG_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    logger.warning("RAG prompt not found. Using default fallback.")
    RAG_SYSTEM_PROMPT = "You are a specialist assistant extracting search queries for clinical guidelines."

# --- SINGLETON VECTOR STORE ---
@lru_cache(maxsize=1)
def get_vectorstore():
    """
    Inicializa a conexão com o ChromaDB.
    """
    print(f"DEBUG: Connecting to VectorStore at: {PERSIST_DIRECTORY}")
    
    embeddings = OpenAIEmbeddings(
        base_url="http://127.0.0.1:1234/v1",
        api_key=SecretStr("lm-studio"),
        check_embedding_ctx_length=False,
        timeout=20.0,
        max_retries=1
    )

    try:
        # Tenta conectar na collection específica
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name="langchain" # Ajustado conforme seu sucesso anterior
        )
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize VectorStore: {e}")
        raise e

# --- MODELO DE QUERY (ULTRA OTIMIZADO) ---
query_llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key=SecretStr("lm-studio"),
    model="meta-llama-3.1-8b-instruct",
    temperature=0,
    seed=SEED,
    timeout=300.0, # Timeout explícito
    max_completion_tokens=60,
    # model_kwargs={"max_tokens": 60},      # FORÇA resposta curta para ser rápida
    max_retries=1
)

def get_smart_fallback_query(risk_analysis: Dict, vitals: Dict) -> str:
    """
    Gera uma query baseada em regras se a LLM falhar.
    Isso garante relevância mesmo sem IA generativa nessa etapa.
    """
    query_parts = ["clinical guidelines"]
    
    # Verifica Scores (Lógica Determinística)
    raw_calcs = risk_analysis.get("calculated_raw", {})
    
    # Verifica NEWS
    news_res = raw_calcs.get("NEWS", {})
    if isinstance(news_res, dict):
        score = news_res.get("total_score")
        if isinstance(score, (int, float)) and score >= 5:
            query_parts.append("Sepsis-3 protocol")
            query_parts.append("clinical deterioration management")
    
    # Verifica Vitais Críticos
    if vitals.get("sbp") and vitals["sbp"] < 90:
        query_parts.append("hypotension management")
    if vitals.get("temperature") and vitals["temperature"] > 38:
        query_parts.append("fever management")
    
    # Se não achou nada específico
    if len(query_parts) == 1:
        return "Standard clinical protocols for sepsis and deterioration"
        
    return " ".join(query_parts)

def filter_vitals_for_prompt(vitals: Dict) -> str:
    if not vitals: return "No vitals available"
    compact = []
    for k, v in vitals.items():
        if v is not None and v != "Alert": 
            compact.append(f"{k}: {v}")
    return ", ".join(compact)

def clinical_rag_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó RAG Otimizado com Smart Fallback.
    """
    logger.info("--- 🔍 NODE: CLINICAL RAG (Safe Mode) ---")
    print("DEBUG: Starting Clinical RAG Node...")

    # 1. Recuperar Contexto
    risk_analysis = state.get("risk_analysis", {})
    risk_text = "Unknown"
    
    if isinstance(risk_analysis, dict):
        risk_text = risk_analysis.get("overall_risk_assessment", "Unknown")
    
    # Fix: Handle both Dict and ScribeSchema object scenarios
    extracted_data = state.get("extracted_data")
    vitals_raw = {}
    
    if hasattr(extracted_data, "extracted_vitals"):
        vitals_raw = extracted_data.extracted_vitals
    elif isinstance(extracted_data, dict):
        vitals_raw = extracted_data.get("extracted_vitals", {})

    # Ensure dict format if it's a model
    if hasattr(vitals_raw, "model_dump"):
        vitals_raw = vitals_raw.model_dump()
    elif hasattr(vitals_raw, "dict"):
        vitals_raw = vitals_raw.dict()

    vitals_compact = filter_vitals_for_prompt(vitals_raw)

    # 2. Query Message Curta
    query_msg = f"""
    Patient: {vitals_compact}
    Risk: {risk_text}
    Task: Create a 5-word search query for clinical guidelines (Sepsis, NEWS2) for this case.
    Output ONLY the query string.
    """

    search_query = ""

    # --- PASSO 1: GERAR QUERY (TENTATIVA LLM) ---
    try:
        print("DEBUG: Generating Search Query via LLM...")
        messages = [
            SystemMessage(content="Output ONLY the search query text."),
            HumanMessage(content=query_msg)
        ]
        
        response = query_llm.invoke(messages)
        
        content_text = response.content
        if isinstance(content_text, list):
            content_text = " ".join([str(item) for item in content_text])
            
        clean_query = content_text.strip().replace('"', '').replace("Search Query:", "").strip()
        
        if clean_query and len(clean_query) > 3:
            search_query = clean_query
            print(f"DEBUG: LLM Search Query Generated: '{search_query}'")

    except Exception as e:
        logger.error(f"RAG Query Generation Failed: {e}")
        print(f"DEBUG: LLM Timeout/Error. Switching to Smart Fallback.")

    # --- PASSO 1.5: SMART FALLBACK (Se LLM falhou ou veio vazio) ---
    if not search_query:
        search_query = get_smart_fallback_query(risk_analysis, vitals_raw)
        print(f"DEBUG: Using Smart Fallback Query: '{search_query}'")

    # --- PASSO 2: BUSCA VETORIAL ---
    retrieved_context = "Guidelines unavailable."
    
    try:
        print("DEBUG: Accessing VectorStore...")
        vectorstore = get_vectorstore()
        
        print(f"DEBUG: Searching documents for: '{search_query}'")
        docs = vectorstore.similarity_search(search_query, k=3)
        
        if docs:
            formatted_docs = []
            for d in docs:
                source = d.metadata.get('source', 'Guideline')
                content = d.page_content[:500].replace("\n", " ") 
                formatted_docs.append(f"[{source}]: {content}...")
            
            retrieved_context = "\n".join(formatted_docs)
            print(f"DEBUG: Retrieval Success ({len(docs)} docs found).")
        else:
            print("DEBUG: Retrieval returned 0 documents.")
            retrieved_context = "No specific guidelines found for this query."

    except Exception as e:
        logger.error(f"VectorStore Retrieval Failed: {e}")
        print(f"CRITICAL WARNING: RAG System failed. Reason: {e}")
        retrieved_context = "NOTICE: RAG System unavailable."

    return {
        "context_text": retrieved_context,
        "rag_query_used": search_query
    }