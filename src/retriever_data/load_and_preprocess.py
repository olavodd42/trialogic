import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
# Sobe dois níveis para chegar na raiz do projeto (ajuste conforme sua pasta)
project_root = os.path.dirname(os.path.dirname(current_dir))
# Define o caminho fixo na raiz
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")
def load_pdf(filepath: str, category: str, source_type: str):
    print(f"📚 Ingerindo: {filepath} como categoria '{category}'...")

    loader = PyPDFLoader(filepath)
    docs = loader.load()

    # 1. Add metadata to the docs
    for doc in docs:
        doc.metadata["category"] = category
        doc.metadata["source_type"] = source_type

    # 2. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"✅ Sucesso! {len(splits)} chunks indexados no ChromaDB.")
