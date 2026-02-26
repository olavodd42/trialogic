import sys
import os
import json
import time
import argparse
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="TriaLogic Batch Processing - Run clinical triage experiments with configurable pipeline components."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.join(os.getcwd(), "results/experiment_results.jsonl"),
        help="Path to the output JSONL file (default: results/experiment_results.jsonl)",
    )
    parser.add_argument(
        "--no-validator",
        action="store_true",
        default=False,
        help="Disable the Validator node (skip physiological plausibility checks).",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        default=False,
        help="Disable Clinical RAG retrieval and Synthesizer nodes.",
    )
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        default=False,
        help="Use LLM-based probabilistic scoring in the Mathematician (default: deterministic calculator).",
    )
    return parser.parse_args()


args = parse_args()

OUTPUT_FILE = args.output
USE_VALIDATOR = not args.no_validator
USE_RAG = not args.no_rag
USE_PROBABILISTIC = args.probabilistic

app = create_workflow(
    use_validator=USE_VALIDATOR,
    use_rag=USE_RAG,
    use_probabilistic=USE_PROBABILISTIC,
)

def main():
    logger.info("=" * 60)
    logger.info("TriaLogic Batch Processing")
    logger.info(f"  Validator:     {'ON' if USE_VALIDATOR else 'OFF'}")
    logger.info(f"  Clinical RAG:  {'ON' if USE_RAG else 'OFF'}")
    logger.info(f"  Mathematician: {'Probabilistic (LLM)' if USE_PROBABILISTIC else 'Deterministic'}")
    logger.info(f"  Output:        {OUTPUT_FILE}")
    logger.info("=" * 60)

    # 1. Load data
    if not os.path.exists(INPUT_CSV):
        logger.error(f"❌ Create the file {INPUT_CSV} first (use filter scripts).")
        return
    
    df = pd.read_csv(INPUT_CSV)
    # df = df.head(20)

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
    logger.info(f"🧪 Start batch experiment with {max(0, len(df) - len(processed_ids))} cases.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 2. Processing loop
    latencies = []
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for row in tqdm(df.to_dict(orient="records"), total=len(df), desc="Processing Agents"):
            start_time = time.time()
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

                # Reporta timings dos agentes
                if hasattr(app, "get_timings"):
                     timings = app.get_timings(final_state)
                     logger.info(f"Tempo por agente: {timings}")

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
                    "risk_analysis": final_state.get("risk_analysis"),
                    "auditor_verdict": final_state.get("auditor_report"),
                    "extracted_vitals": extracted_vitals,
                    "rag_context_used": final_state.get("rag_context_used", USE_RAG),
                    "latency_seconds": None  # placeholder, filled in finally
                }

                logger.info(f"Writing to json: {result_record}")

                f.write(json.dumps(result_record) + "\n")
                f.flush()
            except Exception as e:
                logger.error(f"\n❌Error on ID {subject_id}: {e}")
                error_record = {"subject_id": subject_id, "hadm_id": hadm_id, "error": str(e)}
                f.write(json.dumps(error_record) + "\n")
            finally:
                end_time = time.time()
                case_latency = end_time - start_time
                latencies.append(case_latency)
                # Attach latency to the last written record
                # (record was already flushed, so we log it separately)

    # ── Latency Summary (for TCC / Article) ──
    if latencies:
        arr = np.array(latencies)
        method_label = "TriaLogic"
        if not USE_VALIDATOR:
            method_label += "_NoValidator"
        if not USE_RAG:
            method_label += "_NoRAG"
        if USE_PROBABILISTIC:
            method_label += "_Probabilistic"

        latency_summary = {
            "method": method_label,
            "n_cases": len(arr),
            "total_time_seconds": round(float(np.sum(arr)), 4),
            "mean_seconds": round(float(np.mean(arr)), 4),
            "median_seconds": round(float(np.median(arr)), 4),
            "std_seconds": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
            "min_seconds": round(float(np.min(arr)), 4),
            "max_seconds": round(float(np.max(arr)), 4),
            "p25_seconds": round(float(np.percentile(arr, 25)), 4),
            "p75_seconds": round(float(np.percentile(arr, 75)), 4),
            "p95_seconds": round(float(np.percentile(arr, 95)), 4),
            "p99_seconds": round(float(np.percentile(arr, 99)), 4),
        }
        logger.info("=" * 60)
        logger.info(f"📊 Latency Summary — {method_label}")
        logger.info(f"  N cases:  {latency_summary['n_cases']}")
        logger.info(f"  Total:    {latency_summary['total_time_seconds']:.2f}s")
        logger.info(f"  Mean:     {latency_summary['mean_seconds']:.4f}s")
        logger.info(f"  Median:   {latency_summary['median_seconds']:.4f}s")
        logger.info(f"  Std Dev:  {latency_summary['std_seconds']:.4f}s")
        logger.info(f"  Min/Max:  {latency_summary['min_seconds']:.4f}s / {latency_summary['max_seconds']:.4f}s")
        logger.info(f"  P25/P75:  {latency_summary['p25_seconds']:.4f}s / {latency_summary['p75_seconds']:.4f}s")
        logger.info(f"  P95:      {latency_summary['p95_seconds']:.4f}s")
        logger.info(f"  P99:      {latency_summary['p99_seconds']:.4f}s")
        logger.info("=" * 60)

        # Save latency summary
        latency_path = OUTPUT_FILE.replace('.jsonl', '_latency.json')
        with open(latency_path, 'w', encoding='utf-8') as f:
            json.dump(latency_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 Latency summary saved in {latency_path}")

    print(f"\n✅ Finished experiment. Results saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()