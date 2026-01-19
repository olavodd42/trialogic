import os
import logging
from typing import Dict, Any, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.state.agent_state import AgentState

load_dotenv()

# Logger Configuration
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")

def get_vectorstore():
    """Singleton-like vectorstore retriever."""
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"ChromaDB not found at {PERSIST_DIRECTORY}. Run ingestion first.")
    
    # Tech Lead Note: Mudança para Embeddings Locais (Ollama) para privacidade e custo zero.
    # Recomendado: "nomic-embed-text" ou "mxbai-embed-large"
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    
    # CORREÇÃO CRÍTICA: Mantendo compatibilidade com o banco existente
    return Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embedding_function,
        collection_name="langchain" 
    )

def clinical_rag_node(state: AgentState):
    """
    Agente RAG (Retrieval-Augmented Generation).
    Gera uma query otimizada baseada nos sintomas E NO CONTEXTO CLÍNICO,
    busca documentos e injeta no estado.
    
    Architecture: Context-Aware Query Generator
    """
    logger.info("--- NODE: CLINICAL RAG ---")
    
    # 1. Extração de Inputs com Contexto Expandido
    # Tech Lead Note: Pegamos os primeiros 1500 chars para garantir que a HPI (História da Doença Atual)
    # seja lida. Isso permite identificar 'Bronquiectasia' ou 'MDR organisms' que não estão nos vitais.
    raw_text = state['input'].raw_text[:1500] 
    vitals = state.get("vitals", {})
    risk_score = state.get("risk_score_report", "Risk Score not calculated")
    
    # 2. Configuração da LLM (Query Generator)
    # Tech Lead Note: Usando Llama 3 local via Ollama
    llm = ChatOllama(model="llama3.1", temperature=0)

    # 3. Prompt Hardcoded (A "Constituição" do RAG)
    # Orientador Note: Essas regras garantem que a IA não invente sintomas baseada em viés de treino.
    system_prompt = """
    You are a Semantic Query Optimizer for a Clinical Decision Support System.
    Your goal is to generate a SINGLE search query to retrieve medical protocols.

    INPUT DATA:
    - Patient Vitals (Structured)
    - Clinical Note Snippet (Unstructured Context)
    - Risk Score

    RULES FOR QUERY GENERATION (THE CONSTITUTION):
    1. **FACT CHECK (Vitals Guardrails):** - Look strictly at the provided BP (Blood Pressure) and HR (Heart Rate).
       - If BP is > 90/60 mmHg, **DO NOT** include "hypotension" or "shock" in the query, unless the text explicitly says "dropping BP".
       - If HR is < 100 bpm, **DO NOT** include "tachycardia".
       - If SpO2 is > 94%, **DO NOT** include "hypoxia" or "respiratory failure".
       
    2. **CONTEXT IS KING (Etiology Search):** - Vitals are symptoms, not root causes. 
       - **SCAN THE TEXT** for chronic conditions or specific diagnoses (e.g., "Bronchiectasis", "COPD", "Pneumonia", "Sepsis", "Heart Failure").
       - If found, you **MUST** include the specific condition in the query.
       - *Bad Query:* "Low Oxygen treatment" (Too generic).
       - *Good Query:* "Bronchiectasis exacerbation hypoxia management protocols".

    3. **FORMAT:** Output ONLY the query string. No quotes, no markdown, no explanations.
    """

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
        # Geração da Query
        logger.debug("Generating Search Query via LLM...")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        search_query = response.content.strip()
        logger.info(f"🔍 RAG Query Generated: '{search_query}'")

        # Busca no VectorStore
        logger.debug("Accessing VectorStore...")
        vectorstore = get_vectorstore()
        
        # Tech Lead Tip: Buscamos k=3 para ter redundância, mas sem poluir o contexto.
        docs = vectorstore.similarity_search(search_query, k=3)
        
        if docs:
            formatted_docs = []
            for d in docs:
                source = d.metadata.get('source', 'Guideline')
                # Limpa quebras de linha para economizar tokens no prompt final do Auditor
                content = d.page_content[:600].replace("\n", " ") 
                formatted_docs.append(f"SOURCE [{source}]: {content}...")
            
            retrieved_context = "\n\n".join(formatted_docs)
            logger.info(f"✅ Retrieval Success ({len(docs)} docs found).")
        else:
            logger.warning("⚠️ Retrieval returned 0 documents.")
            retrieved_context = "No specific guidelines found for this query."

    except Exception as e:
        logger.error(f"❌ RAG System Failed: {e}")
        retrieved_context = "CRITICAL: RAG System Offline. Proceed with standard clinical judgment."

    return {
        # "extracted_vitals": vitals,
        "search_query": search_query,
        "rag_context": retrieved_context
    }