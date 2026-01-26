import os
import sys

# Garante que o python encontra os módulos src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa a função nova (note que mudei o nome para ingest_document)
from src.retriever_data.load_and_preprocess import ingest_document

def main():
    # Caminho base (ajuste conforme sua estrutura real)
    base_path = os.getcwd() 
    docs_folder = os.path.join(base_path, "docs")
    
    print("🚀 Iniciando Ingestão de Conhecimento...")

    # 1. Ingerir a "Cheat Sheet" (O Pulo do Gato)
    ingest_document(
        filepath=os.path.join(docs_folder, "definitions.txt"),
        category="protocol_definitions", # Categoria especial
        source_type="gold_standard_definitions"
    )

    # 2. Ingerir os PDFs normais (Exemplos)
    ingest_document(
        filepath=os.path.join(docs_folder, "Sepsis-3.pdf"),
        category="sepsis",
        source_type="guideline"
    )
    
    ingest_document(
        filepath=os.path.join(docs_folder, "NEWS2_Chart.pdf"),
        category="triage",
        source_type="scoring_system"
    )

if __name__ == "__main__":
    main()