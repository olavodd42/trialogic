import uuid
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(filepath: str, category: str, source_type: str):
    pdf_loader = PyPDFLoader(filepath)
    docs = pdf_loader.load()
    for doc in docs:
        doc.metadata["category"] = "cardiology"
        doc.metadata["source_type"] = "clinical_guideline"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "trialogic_pdfs"
    vectorstore = client.get_or_create_collection(name=collection_name)
    vectorstore.add(
        ids=[str(uuid.uuid4()) for _ in splits],
        documents=[split.page_content for split in splits],
        metadatas=[split.metadata for split in splits]
    )