import sys
import os
import json
import pandas as pd
from tqdm import tqdm # Barra de progresso

# Setup de Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import create_workflow
from src.schemas.input_schema import InputSchema
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s]: %(message)s',
    # datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

SEED = 42

# Caminhos
INPUT_CSV = os.path.join(os.getcwd(), "data/gold_standard_dataset.csv") # O dataset filtrado que criamos antes
OUTPUT_FILE = os.path.join(os.getcwd(), "results/norag_experiment_results_v1.jsonl")

app = create_workflow()

def main():
    # 1. Load data
    if not os.path.exists(INPUT_CSV):
        logger.error(f"❌ Create the file {INPUT_CSV} first (use filter scripts).")
        return
    
    df = pd.read_csv(INPUT_CSV)
    
    processed_ids = set()
    
    if os.path.exists(OUTPUT_FILE):
        logger.info(f"📂 Output file found at {OUTPUT_FILE}. Loading processed IDs...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    s_id = str(record.get("subject_id"))
                    h_id = str(record.get("hadm_id")) if record.get("hadm_id") else "None"
                    
                    processed_ids.add((s_id, h_id))
                except json.JSONDecodeError:
                    continue # Pula linhas corrompidas (se houver)

    # df = df.head(5)
    logger.info(f"🧪 Start batch experiment with {len(df) - len(processed_ids)} cases.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 2. Processing loop
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for row in tqdm(df.to_dict(orient="records"), total=len(df), desc="Processing Agents"):
            subject_id = row['subject_id']
            hadm_id = row.get('hadm_id')
            text = row['text']

            check_s_id = str(subject_id)
            check_h_id = str(hadm_id) if pd.notna(hadm_id) else "None"

            if (check_s_id, check_h_id) in processed_ids:
                # Se já existe no set, pulamos silenciosamente para não poluir o log
                continue

            if len(str(text)) < 20:
                continue

            input_obj = InputSchema(
                subject_id=subject_id,
                hadm_id=hadm_id,
                raw_text=text
            )

            try:
                final_state = app.invoke({"input": input_obj})

                # Safely extract vitals whether extracted_data is a dict or Pydantic model
                extracted_data = final_state.get("extracted_data")
                extracted_vitals = None
                if extracted_data:
                    if hasattr(extracted_data, "vitals"):
                         extracted_vitals = extracted_data.vitals.model_dump()
                    elif isinstance(extracted_data, dict):
                         extracted_vitals = extracted_data.get("vitals")

                result_record = {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "cohort": row.get('cohort_type', 'unknown'),
                    "risk_score": final_state.get("risk_score_report"),
                    "auditor_verdict": final_state.get("auditor_report"),
                    "extracted_vitals": extracted_vitals,
                    "rag_context_used": True # Booleano simples se usou contexto
                }

                logger.info(f"Writing to json: {result_record}")

                f.write(json.dumps(result_record) + "\n")
                f.flush()
            except Exception as e:
                logger.error(f"\n❌Error on ID {subject_id}: {e}")
                error_record = {"subject_id": subject_id, "hadm_id": hadm_id, "error": str(e)}
                f.write(json.dumps(error_record) + "\n")

    print(f"\n✅ Finished experiment. Results saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()