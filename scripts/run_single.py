"""Run a single clinical note through the TriaLogic pipeline for quick testing."""

import argparse
import logging

from src.main import create_workflow
from src.schemas.input_schema import InputSchema

# Configure logging for terminal output
logging.basicConfig(level=logging.INFO)

# A known Sepsis case (should activate RAG if Sepsis docs are ingested).
MOCK_TEXT = """
Triage Note: 55y male presents with fever and hypotension.
History of UTI. 
Vitals: T 38.9, HR 115, BP 85/50, RR 22, SpO2 94% on RA.
Alert but lethargic.
"""

def test_rag_integration():
    """Run a single mock case and display the pipeline output."""
    parser = argparse.ArgumentParser(description="TriaLogic Single Case Runner")
    parser.add_argument("--no-validator", action="store_true", help="Disable Validator node.")
    parser.add_argument("--no-rag", action="store_true", help="Disable Clinical RAG and Synthesizer.")
    parser.add_argument("--probabilistic", action="store_true", help="Use LLM-based probabilistic Mathematician.")
    args = parser.parse_args()

    print("Starting single-case pipeline test...")
    app = create_workflow(
        use_validator=not args.no_validator,
        use_rag=not args.no_rag,
        use_probabilistic=args.probabilistic,
    )
    
    input_data = InputSchema(
        subject_id=1,
        hadm_id=1,
        raw_text=MOCK_TEXT
    )
    
    result = app.invoke({"input": input_data})
    
    print("\n" + "="*40)
    print("TEST RESULT")
    print("="*40)
    
    # 1. Check RAG
    rag_used = result.get("rag_context_used", False)
    auditor = result.get("auditor_report", {})
    quote = auditor.get("evidence_quote", "")
    
    print(f"1. RAG enabled? {'Yes' if rag_used else 'No'}")
    
    if len(str(quote)) > 10:
        print(f"2. Quote found: \"{quote[:100]}...\"")
    else:
        print(f"2. Quote: Empty (RAG ran but found no docs, or failed silently).")
        
    # 3. Check diagnosis
    print(f"3. Verdict: {auditor.get('clinical_risk_category')}")
    print(f"4. Suggestion: {auditor.get('clinical_suggestion')}")

if __name__ == "__main__":
    test_rag_integration()