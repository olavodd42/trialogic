import sys
import os
import json
import polars as pl
from tqdm import tqdm # Barra de progresso

# Setup de Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import create_workflow
from src.schemas.input_schema import InputSchema

# Caminhos
INPUT_CSV = os.path.join(os.getcwd(), "data/gold_standard_dataset.csv") # O dataset filtrado que criamos antes
OUTPUT_FILE = os.path.join(os.getcwd(), "results/experiment_results_v1.jsonl")

app = create_workflow()

def main():
    # 1. Load data
    if not os.path.exists(INPUT_CSV):
        print(f"Erro: Create the file {INPUT_CSV} first (use filter scripts).")
        return
    
    df = pl.read_csv(INPUT_CSV)
    df = df[0]
    print(f"🧪 Start batch experiment with {len(df)} cases.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 2. Processing loop
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Processing Agents"):
            subject_id = row['subject_id']
            text = row['text']

            if len(text) < 50:
                continue

            input_obj = InputSchema(
                subject_id=subject_id,
                hadm_id=row.get('hadm_id'),
                raw_text=text
            )

            try:
                final_state = app.invoke({"input": input_obj})

                # Safely extract vitals whether extracted_data is a dict or Pydantic model
                extracted_data = final_state.get("extracted_data")
                extracted_vitals = None
                if extracted_data:
                    if hasattr(extracted_data, "clinical"):
                         # Pydantic model access
                         extracted_vitals = extracted_data.clinical.vitals.model_dump()
                    elif isinstance(extracted_data, dict):
                         # Dictionary access
                         extracted_vitals = extracted_data.get("clinical", {}).get("vitals")

                result_record = {
                    "subject_id": subject_id,
                    "cohort": row.get('cohort_type', 'unknown'),
                    "risk_score": final_state.get("risk_score_report"),
                    "auditor_verdict": final_state.get("auditor_report"),
                    "extracted_vitals": extracted_vitals,
                    "rag_context_used": len(final_state.get("context_text", "")) > 10 # Booleano simples se usou contexto
                }

                f.write(json.dumps(result_record) + "\n")
                f.flush()
            except Exception as e:
                print(f"\nEror on ID {subject_id}: {e}")
                # Logar erro no arquivo também para não perder rastro
                error_record = {"subject_id": subject_id, "error": str(e)}
                f.write(json.dumps(error_record) + "\n")

    print(f"\n✅ Finished experiment. Results saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()