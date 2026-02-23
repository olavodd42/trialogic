import os
import logging
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")

# Module-level cache to avoid re-loading the embedding model on every call
_vectorstore_cache = None

def get_vectorstore():
    """Singleton vectorstore retriever — loads embedding model once."""
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    if not os.path.exists(PERSIST_DIRECTORY):
        logger.error("ChromaDB not found.")
        raise FileNotFoundError(f"ChromaDB not found at {PERSIST_DIRECTORY}. Run ingestion first.")
    
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    _vectorstore_cache = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embedding_function,
        collection_name="langchain" 
    )
    return _vectorstore_cache