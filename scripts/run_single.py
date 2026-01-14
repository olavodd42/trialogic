import sys
import os
from pprint import pprint
from typing import cast
import pandas as pd

sys.path.append(os.getcwd())

from src.main import create_workflow
from src.schemas.input_schema import InputSchema
from src.state.agent_state import AgentState

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))
# DATA_PATH = os.path.join(project_root, "data/gold_standard_dataset.csv")

# messages = pd.read_csv(DATA_PATH)

# Determine clinical case (Fake example)
fake_note = r"""
Patient: John Doe, 65M.
Chief Complaint: Fever and confusion.
HPI: Patient brought by wife due to altered mental status starting this morning. 
Complains of chills. History of UTI.
Vitals:
- HR: 115 bpm
- BP: 85/50 mmHg
- Temp: 39.2 C
- RR: 24 rpm
- SpO2: 94% on RA
Physical Exam: Warm extremities, disoriented to time/place.
"""

app = create_workflow()

def main():
    print("🚀 Starting TriaLogic Single Case Test...")

    # 2. Create input object
    patient_input = InputSchema(
        subject_id=12345,     # ID Fictício
        hadm_id=None,         # Ainda não internou
        raw_text=fake_note
    )

    # 3. Invoke the graph
    initial_state = cast(AgentState, {
        "input": patient_input,
        "extracted_data": None,
        "validation_errors": [],
        "validation_messages": [],
        "attempts": 0,
        "risk_score_report": None,
        "search_query": None,
        "context_category": None,
        "context_text": None,
        "auditor_report": None,
        "next_step": None,
        "plan": None
        })
    
    try:
        final_state = app.invoke(initial_state)

        print("\n" + "="*50)
        print("✅ FINISHED EXECUTION")
        print("="*50)

        print(f"\n🩺 Calculated Report:\n{final_state.get('risk_score_report')}")
        
        print(f"\n📚 Query RAG used:\n{final_state.get('search_query')}")
        
        print(f"\n⚖️ Auditor (Synthesizer) Report:")
        pprint(final_state.get('auditor_report'))

    except Exception as e:
        print(f"❌ Error during execution: {e}")

if __name__ == '__main__':
    main()