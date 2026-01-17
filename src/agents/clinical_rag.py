"""
Clinical RAG Agent Module - Robust Version (TCC Safe).

Implements Circuit Breaker patterns to prevent system hangs during
local inference (LM Studio) or vector store locking.
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
from langchain_core.documents import Document
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

# --- SINGLETON VECTOR STORE (COM TIMEOUT) ---
@lru_cache(maxsize=1)
def get_vectorstore():
    """
    Inicializa a conexão com o ChromaDB de forma segura.
    ATENÇÃO: Requer que o LM Studio esteja servindo embeddings ou falhará graciosamente.
    """
    logger.info("Initializing VectorStore Connection...")
    
    # Configuração com Timeout para evitar travamento eterno se o LM Studio não responder
    embeddings = OpenAIEmbeddings(
        base_url="http://127.0.0.1:1234/v1",
        api_key=SecretStr("lm-studio"),
        check_embedding_ctx_length=False,
        timeout=10.0, # Timeout curto para falhar rápido (Fail Fast)
        max_retries=1         # Não tentar infinitamente
    )

    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name="clinical_guidelines"
        )
        logger.info("VectorStore initialized successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize VectorStore: {e}")
        raise e

# --- MODELO DE QUERY (COM TIMEOUT) ---
# Usamos uma instância separada com timeout agressivo para evitar hangs
query_llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key=SecretStr("lm-studio"),
    model="meta-llama-3.1-8b-instruct",
    temperature=0,
    seed=SEED,
    timeout=30.0, # 30s max para gerar a query
    max_retries=1
)

def clinical_rag_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó RAG com proteção 'Circuit Breaker'.
    Se qualquer etapa falhar (LLM ou VectorDB), ele retorna vazio mas NÃO TRAVA.
    """
    logger.info("--- 🔍 NODE: CLINICAL RAG (Safe Mode) ---")
    print("DEBUG: Starting Clinical RAG Node...")

    # 1. Recuperar Contexto do Agente Matemático
    risk_analysis = state.get("risk_analysis", {})
    # Fallback: Se o mathematician falhou e retornou string
    if isinstance(risk_analysis, dict):
        risk_text = risk_analysis.get("overall_risk_assessment", "Unknown Risk")
        calculated_raw = risk_analysis.get("calculated_raw", {})
    else:
        risk_text = str(risk_analysis)
        calculated_raw = {}

    vitals = state.get("extracted_data", {}).get("extracted_vitals", {})

    # Construir mensagem para gerar a Query de Busca
    query_msg = f"""
    Based on the following patient status, generate a specialized search query 
    to retrieve clinical guidelines (Sepsis-3, NEWS2, ACLS).
    
    PATIENT VITALS: {vitals}
    RISK ASSESSMENT: {risk_text}
    SCORES: {calculated_raw}
    
    Output ONLY the search query string. No quotes, no preamble.
    Example: "Sepsis-3 protocol for hypotension and fever"
    """

    search_query = "Standard clinical protocols for sepsis and deterioration" # Default Fallback

    # --- PASSO 1: GERAR QUERY (COM PROTEÇÃO) ---
    try:
        print("DEBUG: Generating Search Query via LLM...")
        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=query_msg)
        ]
        
        response = query_llm.invoke(messages)
        
        # Handle potential list content (LangChain multimodal support)
        content = response.content
        if not isinstance(content, str):
            content = str(content)
            
        raw_query = content.strip()
        
        # Limpeza básica da resposta
        clean_query = raw_query.replace('"', '').replace("Search Query:", "").strip()
        if clean_query:
            search_query = clean_query
            
        print(f"DEBUG: Search Query Generated: '{search_query}'")

    except Exception as e:
        logger.error(f"RAG Query Generation Failed: {e}")
        print("DEBUG: LLM Timeout/Error. Using default query.")
        # Não retornamos erro, seguimos com a query padrão para tentar salvar o processo.

    # --- PASSO 2: BUSCA VETORIAL (COM PROTEÇÃO) ---
    retrieved_context = "Guidelines unavailable (Retrieval System Error)."
    
    try:
        print("DEBUG: Accessing VectorStore...")
        vectorstore = get_vectorstore()
        
        # Busca com filtro de confiança
        print(f"DEBUG: Searching documents for: '{search_query}'")
        docs = vectorstore.similarity_search(search_query, k=3)
        
        if docs:
            formatted_docs = []
            for d in docs:
                source = d.metadata.get('source', 'Guideline')
                formatted_docs.append(f"[{source}]: {d.page_content}")
            
            retrieved_context = "\n\n".join(formatted_docs)
            print(f"DEBUG: Retrieval Success ({len(docs)} docs found).")
        else:
            print("DEBUG: Retrieval returned 0 documents.")
            retrieved_context = "No specific guidelines found for this query."

    except Exception as e:
        # AQUI É ONDE O CÓDIGO TRAVAVA (Possível falha de Embeddings)
        logger.error(f"VectorStore Retrieval Failed: {e}")
        print(f"CRITICAL WARNING: RAG System failed. Reason: {e}")
        print("Continuing workflow without RAG context (Degraded Mode).")
        # Definimos um contexto de fallback para que o Auditor ainda possa funcionar
        retrieved_context = "NOTICE: RAG System unavailable. Proceed using standard internal knowledge of Sepsis-3 and NEWS2."

    # Retorna o contexto para o estado
    return {
        "context_text": retrieved_context,
        "rag_query_used": search_query
    }