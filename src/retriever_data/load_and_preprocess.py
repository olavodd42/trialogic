"""Document ingestion pipeline for the ChromaDB vector store."""

import logging
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logger = logging.getLogger(__name__)

# Directory configuration
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

def ingest_document(filepath: str, category: str, source_type: str):
    """
    Ingests a document into the ChromaDB vector store.
    
    Loads a document from the specified filepath (supports PDF and TXT), 
    assigns metadata (category, source_type), splits it into chunks, 
    generates embeddings, and stores them in the persistent ChromaDB directory.

    Args:
        filepath (str): Absolute or relative path to the document file.
        category (str): A label to categorize the document (e.g., 'protocol', 'guideline').
        source_type (str): The origin or type of the source (e.g., 'internal', 'external').

    Returns:
        None
    """
    logger.debug("Ingesting: %s ...", filepath)

    # 1. Detecting loading type
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        logger.error("Format not supported: %s", filepath)
        return

    # 2. Loading
    try:
        docs = loader.load()
    except Exception as e:
        logger.error("Error reading file: %s", e)
        return

    # 3.Metadata
    for doc in docs:
        doc.metadata["category"] = category
        doc.metadata["source_type"] = source_type
        if "definitions.txt" in filepath:
            doc.metadata["priority"] = "high"

    # 4. Split
    chunk_size = 500 if filepath.endswith(".txt") else 1000
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 5. Embeddings Configuration
    # Using sentence-transformers locally (compatible with LLaMA ecosystem)
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 6. Indexing on ChromaDB
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    logger.info("Success! %d chunks of '%s' indexed.", len(splits), category)