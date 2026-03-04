"""Batch-process clinical notes through the TriaLogic multi-agent pipeline."""

import sys
import os
import json
import time
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bar

# Path setup
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

# Paths
INPUT_CSV = os.path.join(os.getcwd(), "data/gold_standard_dataset.csv")  # Filtered dataset created earlier


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
    logger.info("  Validator:     %s", 'ON' if USE_VALIDATOR else 'OFF')
    logger.info("  Clinical RAG:  %s", 'ON' if USE_RAG else 'OFF')
    logger.info("  Mathematician: %s", 'Probabilistic (LLM)' if USE_PROBABILISTIC else 'Deterministic')
    logger.info("  Output:        %s", OUTPUT_FILE)
    logger.info("=" * 60)

    # 1. Load data
    if not os.path.exists(INPUT_CSV):
        logger.error("Create the file %s first (use filter scripts).", INPUT_CSV)
        return
    
    df = pd.read_csv(INPUT_CSV)
    # df = df.head(20)

    processed_ids = set()
    
    if os.path.exists(OUTPUT_FILE):
        logger.info("Output file found at %s. Loading processed IDs...", OUTPUT_FILE)
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    s_id = str(record.get("subject_id"))
                    h_id = str(record.get("hadm_id")) if record.get("hadm_id") else "None"
                    
                    processed_ids.add((s_id, h_id))
                except json.JSONDecodeError:
                    continue  # Skip corrupted lines (if any)

    # df = df.head(5)
    logger.info("Start batch experiment with %d cases.", max(0, len(df) - len(processed_ids)))

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
                # Already in set, skip silently to avoid log pollution
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

                # Report per-agent timings
                if hasattr(app, "get_timings"):
                     timings = app.get_timings(final_state)
                     logger.info("Per-agent timing: %s", timings)

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

                logger.info("Writing to json: %s", result_record)

                f.write(json.dumps(result_record) + "\n")
                f.flush()
            except Exception as e:
                logger.error("Error on ID %s: %s", subject_id, e)
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
        logger.info("Latency Summary -- %s", method_label)
        logger.info("  N cases:  %d", latency_summary['n_cases'])
        logger.info("  Total:    %.2fs", latency_summary['total_time_seconds'])
        logger.info("  Mean:     %.4fs", latency_summary['mean_seconds'])
        logger.info("  Median:   %.4fs", latency_summary['median_seconds'])
        logger.info("  Std Dev:  %.4fs", latency_summary['std_seconds'])
        logger.info("  Min/Max:  %.4fs / %.4fs", latency_summary['min_seconds'], latency_summary['max_seconds'])
        logger.info("  P25/P75:  %.4fs / %.4fs", latency_summary['p25_seconds'], latency_summary['p75_seconds'])
        logger.info("  P95:      %.4fs", latency_summary['p95_seconds'])
        logger.info("  P99:      %.4fs", latency_summary['p99_seconds'])
        logger.info("=" * 60)

        # Save latency summary
        latency_path = OUTPUT_FILE.replace('.jsonl', '_latency.json')
        with open(latency_path, 'w', encoding='utf-8') as f:
            json.dump(latency_summary, f, indent=2, ensure_ascii=False)
        logger.info("Latency summary saved in %s", latency_path)

    print(f"\nFinished experiment. Results saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()