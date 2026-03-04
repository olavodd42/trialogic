"""Ingest clinical guideline documents into the ChromaDB vector store."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever_data.load_and_preprocess import ingest_document

def main():
    # Base path (adjust according to your actual project structure)
    base_path = os.getcwd() 
    docs_folder = os.path.join(base_path, "docs")
    
    print("Starting knowledge ingestion...")

    # 1. Ingest the definitions cheat sheet
    ingest_document(
        filepath=os.path.join(docs_folder, "definitions.txt"),
        category="protocol_definitions", # Categoria especial
        source_type="gold_standard_definitions"
    )

    # 2. Ingest PDF guidelines
    ingest_document(
        filepath=os.path.join(docs_folder, "Sepsis-3.pdf"),
        category="sepsis",
        source_type="guideline"
    )

    ingest_document(
        filepath=os.path.join(docs_folder, "Cardio.pdf"),
        category="chest pain",
        source_type="guideline"
    )

    # ingest_document(
    #     filepath=os.path.join(docs_folder, "ChestPain.pdf"),
    #     category="chest pain",
    #     source_type="guideline"
    # )

    ingest_document(
        filepath=os.path.join(docs_folder, "Trauma.pdf"),
        category="trauma",
        source_type="guideline"
    )

if __name__ == "__main__":
    main()