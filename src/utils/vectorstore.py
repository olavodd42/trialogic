import os
import logging
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")

def get_vectorstore():
    """Singleton-like vectorstore retriever."""
    if not os.path.exists(PERSIST_DIRECTORY):
        logger.error("ChromaDB not found.")
        raise FileNotFoundError(f"ChromaDB not found at {PERSIST_DIRECTORY}. Run ingestion first.")
    
    from langchain_openai import OpenAIEmbeddings
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )
    
    return Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embedding_function,
        collection_name="langchain" 
    )