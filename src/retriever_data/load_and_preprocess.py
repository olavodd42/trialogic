import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

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
    embedding_function = OpenAIEmbeddings(
        base_url="http://localhost:1234/v1",  # Aponta para o LM Studio
        api_key=SecretStr("lm-studio"),                  # Obrigatório, mas pode ser qualquer string
        model="text-embedding-nomic-embed-text-v1.5",        # O nome deve bater com o ID no LM Studio (ou use "text-embedding-ada-002" como placeholder se der erro)
        check_embedding_ctx_length=False      # Impede validação de token que falha localmente
    )
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"✅ Sucesso! {len(splits)} chunks indexados no ChromaDB.")
