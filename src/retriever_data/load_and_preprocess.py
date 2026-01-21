import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
import logging

logger = logging.getLogger(__name__)

# Configuração de Diretório
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

def ingest_document(filepath: str, category: str, source_type: str):
    """
    Função universal de ingestão. Detecta se é PDF ou TXT e processa.
    """
    logger.debug(f"📚 Ingesting: {filepath}...")

    # 1. Detecting loading type
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        logger.error(f"❌ Format not supported: {filepath}")
        return

    # 2. Loading
    try:
        docs = loader.load()
    except Exception as e:
        logger.error(f"❌ Erro ao ler arquivo: {e}")
        return

    # 3.Metadata
    for doc in docs:
        doc.metadata["category"] = category
        doc.metadata["source_type"] = source_type
        # Se for o definitions.txt, damos um boost artificial na relevância via metadado (opcional)
        if "definitions.txt" in filepath:
            doc.metadata["priority"] = "high"

    # 4. Split
    chunk_size = 500 if filepath.endswith(".txt") else 1000
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 5. Embeddings Configuration
    embedding_function = OpenAIEmbeddings(
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("lm-studio"),      
        check_embedding_ctx_length=False
    )

    # 6. Indexing on ChromaDB
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    logger.info(f"✅ Success! {len(splits)} chunks of '{category}' indexed.")