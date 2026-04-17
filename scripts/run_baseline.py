"""Baseline zero-shot experiment: single LLM call per clinical note."""

import logging
import os
import json
import re
import time
import argparse
import threading
import numpy as np
import pandas as pd
import httpx
import subprocess
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Per-case timeout (seconds) ──
LLM_TIMEOUT = 180
# ── Max consecutive failures before aborting ──
MAX_CONSECUTIVE_FAILURES = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline Zero-Shot - Run clinical triage baseline experiment."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.join(BASE_DIR, "results/baseline_results.jsonl"),
        help="Path to the output JSONL file (default: results/baseline_results.jsonl)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.join(BASE_DIR, "data/gold_standard_dataset.csv"),
        help="Path to the input CSV dataset (default: data/gold_standard_dataset.csv)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of cases to process (default: all)",
    )
    return parser.parse_args()

# 1. Configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Silence verbose HTTP debug logs from httpx/httpcore
for noisy_logger in ("httpx", "httpcore", "httpcore.http11", "httpcore.connection"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

def create_llm():
    """Create a fresh ChatOllama instance."""
    return ChatOllama(
        model="llama3.1:latest",
        temperature=0,
        seed=42,
        num_predict=2048,   # Cap output tokens; large enough for full JSON + preamble
        keep_alive=-1,      # Keep model loaded in VRAM between calls
    )

llm = create_llm()


def invoke_with_hard_timeout(messages, timeout_seconds: int = LLM_TIMEOUT):
    """Invoke LLM with a hard wall-clock timeout using a daemon thread.

    If the call doesn't finish in time, the daemon thread is abandoned,
    the LLM client is recreated (new connection), and TimeoutError is raised.
    """
    global llm
    result = [None]
    error = [None]

    def _invoke():
        try:
            result[0] = llm.invoke(messages)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_invoke, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        # Thread still running — abandon it and recreate client
        logger.warning("Hard timeout (%ds) exceeded. Recreating LLM client...", timeout_seconds)
        llm = create_llm()
        raise TimeoutError(f"LLM call exceeded {timeout_seconds}s hard timeout")

    if error[0]:
        raise error[0]

    return result[0]

# 2. Monolythic Prompt
BASELINE_PROMPT = """
You are a clinical AI. Your task is to Extract vitals and Calculate risk scores.

INPUT TEXT:
{clinical_text}

INSTRUCTIONS:
1. Extract: HR, SBP, DBP, RR, O2Sat, Temp, AVPU, Supplemental Oxygen.
2. [CRITICAL] Normalize: For temperature, if it is in Fahrenheit, convert to Celsius.
3. Calculate:
   - NEWS2 Score (National Early Warning Score 2)
   - MEWS Score (Modified Early Warning Score)
   *Note: If mental status is confused/disoriented, count it as AVPU score > 0.*

OUTPUT FORMAT:
Return ONLY a valid JSON object. No intro text. No markdown formatting like ```json.
DO NOT ADD COMMENTS (like //) inside the JSON.

{{
  "text_report": "SCORE TOTAL NEWS: <insert number> SCORE TOTAL MEWS: <insert number>",
  "extracted_vitals": {{
    "heartrate": <int or null>,
    "resprate": <int or null>,
    "temperature": <float or null>,
    "o2sat": <int or null>,
    "sbp": <int or null>,
    "dbp": <int or null>,
    "avpu": "<Alert|Voice|Pain|Unresponsive|Confusion>",
    "supplemental_oxygen": <true|false>
  }}
}}
"""

def robust_json_extractor(text: str) -> Optional[Dict]:
    """Extract a JSON object from LLM output, handling preamble, code blocks,
    and stray braces from echoed prompt content."""
    EXPECTED_KEYS = {"extracted_vitals", "text_report"}
    text = re.sub(r'//.*', '', text)  # strip JS-style comments

    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown code blocks
    clean = re.sub(r'```(?:json|JSON)?\s*', '', text)
    clean = re.sub(r'```\s*', '', clean)
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Brace-counting: find ALL complete JSON objects, pick best one
    candidates = []
    i = 0
    while i < len(clean):
        if clean[i] != '{':
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(clean)):
            c = clean[j]
            if esc:
                esc = False
                continue
            if c == '\\':
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    candidate_str = clean[i:j+1]
                    if len(candidate_str) > 10:
                        try:
                            obj = json.loads(candidate_str)
                            if isinstance(obj, dict):
                                candidates.append(obj)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    break
        i += 1

    if not candidates:
        # 4. Last resort: greedy regex with newline flattening
        match = re.search(r'(\{.*\})', clean, re.DOTALL)
        if match:
            json_str = re.sub(r'\n', ' ', match.group(1))
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    # Prefer candidate with expected keys
    for c in candidates:
        if EXPECTED_KEYS & set(c.keys()):
            return c
    return max(candidates, key=lambda c: len(json.dumps(c)))

def load_completed_hadm_ids(output_path: str) -> set:
    """Load hadm_ids already processed from the output JSONL file."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    hadm_id = entry.get('hadm_id')
                    if hadm_id is not None:
                        completed.add(hadm_id)
                except json.JSONDecodeError:
                    continue
    return completed


def check_ollama_health(base_url: str = "http://127.0.0.1:11434") -> bool:
    """Check if Ollama is reachable."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_ollama(base_url: str = "http://127.0.0.1:11434", max_wait: int = 60):
    """Wait until Ollama is reachable, optionally trying to start it."""
    logger.info("Waiting for Ollama to become available...")
    for i in range(max_wait // 5):
        if check_ollama_health(base_url):
            logger.info("Ollama is available.")
            return True
        if i == 0:
            # Try starting ollama serve in background
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                logger.info("Attempted to start 'ollama serve'...")
            except FileNotFoundError:
                logger.warning("ollama binary not found in PATH.")
        time.sleep(5)
    logger.error("Ollama did not become available.")
    return False


def process_baseline_batch(df: pd.DataFrame, limit: Optional[int] = None, output_path: Optional[str] = None):
    latencies = []
    results_count = 0
    errors_count = 0
    
    # 3. Limit for fast test if necessary
    if limit:
        df = df.head(limit)

    # Skip already processed cases
    completed_ids = load_completed_hadm_ids(output_path) if output_path else set()
    if completed_ids:
        before = len(df)
        df = df[~df['hadm_id'].isin(completed_ids)]
        skipped = before - len(df)
        logger.info("Skipping %d already processed cases (%d found in output file).", skipped, len(completed_ids))

    # Verify Ollama is available before starting
    if not check_ollama_health():
        if not wait_for_ollama():
            logger.error("Aborting: Ollama is not running.")
            return {}

    logger.info("Starting baseline (Vanilla Llama 3.1) in %d cases...", len(df))

    # Open output file in append mode for incremental saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if output_path else None
    out_file = open(output_path, 'a', encoding='utf-8') if output_path else None

    consecutive_failures = 0

    try:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            start_time = time.time()
            text = str(row.get('text', ''))

            if len(text) < 15:
                continue
            
            result_entry = None
            try:
                # Check if too many consecutive failures
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.warning("%d consecutive failures. Checking Ollama health...", MAX_CONSECUTIVE_FAILURES)
                    if not wait_for_ollama():
                        logger.error("Aborting batch: Ollama is unreachable.")
                        break
                    consecutive_failures = 0  # Reset after recovery

                # Zero-shot inference (with hard wall-clock timeout)
                prompt = BASELINE_PROMPT.format(clinical_text=text)
                response = invoke_with_hard_timeout([
                    HumanMessage(content=prompt)
                ])
                
                raw_content = response.content

                logger.debug(raw_content)

                # Try to parse the response
                parsed = robust_json_extractor(raw_content)
                
                if parsed:
                    result_entry = {
                        "hadm_id": row.get('hadm_id'),
                        "subject_id": row.get('subject_id'),
                        "cohort": row.get('cohort_type', 'sepsis'),
                        "extracted_vitals": parsed.get("extracted_vitals", {}),
                        "risk_score": parsed.get("text_report", ""),
                        "method": "baseline_zero_shot",
                        "latency_seconds": None
                    }
                    results_count += 1
                    consecutive_failures = 0  # Reset on success
                else:
                    logger.warning("Failed to parse JSON for ID %s", row.get('subject_id'))
                    result_entry = {
                        "hadm_id": row.get('hadm_id'),
                        "error": "JSON_PARSE_ERROR",
                        "raw_output": str(raw_content)[:500]
                    }
                    errors_count += 1
                    # Parse errors are not connection failures
                    consecutive_failures = 0

            except (httpx.TimeoutException, httpx.ConnectError, ConnectionError, OSError) as e:
                consecutive_failures += 1
                logger.warning("Connection/Timeout error on index %s (hadm_id=%s): %s. Consecutive failures: %d/%d", index, row.get('hadm_id'), type(e).__name__, consecutive_failures, MAX_CONSECUTIVE_FAILURES)
                result_entry = {
                    "hadm_id": row.get('hadm_id'),
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                    "method": "baseline_error"
                }
                errors_count += 1

            except Exception as e:
                consecutive_failures += 1
                logger.error("Error on index %s: %s", index, e)
                result_entry = {
                    "hadm_id": row.get('hadm_id'),
                    "error": f"EXCEPTION: {str(e)}",
                    "method": "baseline_error"
                }
                errors_count += 1

            finally:
                end_time = time.time()
                case_latency = end_time - start_time
                latencies.append(case_latency)
                # Attach latency and flush to disk immediately
                if result_entry is not None:
                    if "latency_seconds" in result_entry:
                        result_entry["latency_seconds"] = round(case_latency, 4)
                    if out_file:
                        out_file.write(json.dumps(result_entry) + '\n')
                        out_file.flush()
    finally:
        if out_file:
            out_file.close()

    logger.info("Processed %d OK, %d errors.", results_count, errors_count)

    # ── Latency Summary (for TCC / Article) ──
    latency_summary = compute_latency_summary(latencies, method="baseline_zero_shot")
    return latency_summary

def compute_latency_summary(latencies: list, method: str = "baseline") -> Dict[str, Any]:
    """Compute comprehensive latency statistics suitable for academic reporting."""
    if not latencies:
        return {}
    arr = np.array(latencies)
    summary = {
        "method": method,
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
    logger.info("Latency Summary -- %s", method)
    logger.info("  N cases:  %d", summary['n_cases'])
    logger.info("  Total:    %.2fs", summary['total_time_seconds'])
    logger.info("  Mean:     %.4fs", summary['mean_seconds'])
    logger.info("  Median:   %.4fs", summary['median_seconds'])
    logger.info("  Std Dev:  %.4fs", summary['std_seconds'])
    logger.info("  Min/Max:  %.4fs / %.4fs", summary['min_seconds'], summary['max_seconds'])
    logger.info("  P25/P75:  %.4fs / %.4fs", summary['p25_seconds'], summary['p75_seconds'])
    logger.info("  P95:      %.4fs", summary['p95_seconds'])
    logger.info("  P99:      %.4fs", summary['p99_seconds'])
    logger.info("=" * 60)
    return summary


if __name__ == "__main__":
    args = parse_args()

    CSV_PATH = args.input
    OUTPUT_PATH = args.output

    logger.info("=" * 60)
    logger.info("Baseline Zero-Shot Experiment")
    logger.info("  Input:   %s", CSV_PATH)
    logger.info("  Output:  %s", OUTPUT_PATH)
    logger.info("  Limit:   %s", args.limit or 'all')
    logger.info("=" * 60)
    
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if 'hadm_id' in df.columns:
            df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce')
        
        latency_summary = process_baseline_batch(df, limit=args.limit, output_path=OUTPUT_PATH)
        
        # Save latency summary (JSON) for article/TCC
        latency_path = OUTPUT_PATH.replace('.jsonl', '_latency.json')
        with open(latency_path, 'w', encoding='utf-8') as f:
            json.dump(latency_summary, f, indent=2, ensure_ascii=False)
        logger.info("Latency summary saved in %s", latency_path)
                
        logger.info("Baseline finished. Results saved in %s", OUTPUT_PATH)
    else:
        logger.error("Dataset not found: %s", CSV_PATH)
