import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever_data.load_and_preprocess import load_pdf

DOCS_DIR = os.path.join(os.getcwd(), "docs")

def main():
    print("🚀 Starting knowledge ingestion...")

    # Ingest protocols
    load_pdf(
        filepath=os.path.join(DOCS_DIR, "Sepsis-3.pdf"),
        category="sepsis",
        source_type="guideline"
    )

    load_pdf(
        filepath=os.path.join(DOCS_DIR, "NEWS2_Chart.pdf"),
        category="triage",
        source_type="scoring_system"
    )

if __name__ == "__main__":
    main()