"""Data-integrity audit comparing ground truth CSV with experiment JSONL results."""

import json
import os

import pandas as pd

# Configuration
CSV_PATH = "data/gold_standard_dataset.csv"
JSONL_PATH = "results/experiment_results_v1.jsonl"

def audit_discrepancy():
    print("Starting data integrity audit...\n")

    # 1. Load Ground Truth
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return
    df_gt = pd.read_csv(CSV_PATH)
    total_gt = len(df_gt)
    # Normalise IDs to string to ensure matching
    gt_ids = set(df_gt['subject_id'].astype(str) + "_" + df_gt['hadm_id'].astype(str))
    
    print(f"Total cases in ground truth (CSV): {total_gt}")

    # 2. Load Experiment Results
    if not os.path.exists(JSONL_PATH):
        print(f"Error: JSONL not found at {JSONL_PATH}")
        return
    
    processed_rows = []
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Check for explicit error record
                if "error" in data:
                    processed_rows.append({"type": "error", "id": f"{data.get('subject_id')}_{data.get('hadm_id')}"})
                else:
                    processed_rows.append({"type": "success", "id": f"{data.get('subject_id')}_{data.get('hadm_id')}"})
            except:
                pass

    total_processed = len(processed_rows)
    success_ids = {row['id'] for row in processed_rows if row['type'] == 'success'}
    error_ids = {row['id'] for row in processed_rows if row['type'] == 'error'}
    
    print(f"Total cases in results (JSONL): {total_processed}")
    print(f"   Successes (vitals extracted): {len(success_ids)}")
    print(f"   Explicit errors (exceptions): {len(error_ids)}")

    # 3. GAP Analysis (Who went missing?)
    processed_ids = success_ids.union(error_ids)
    missing_ids = gt_ids - processed_ids
    
    print("\n" + "="*40)
    print(f"SKIPPED RECORDS: {len(missing_ids)}")
    print("="*40)

    if len(missing_ids) > 0:
        print("\nInvestigating cause of missing records (sample):")
        # Check the original CSV to see if they were short texts
        df_gt['unique_id'] = df_gt['subject_id'].astype(str) + "_" + df_gt['hadm_id'].astype(str)
        missing_df = df_gt[df_gt['unique_id'].isin(missing_ids)]
        
        short_text_count = 0
        for _, row in missing_df.iterrows():
            text_len = len(str(row['text']))
            if text_len < 50:
                short_text_count += 1
            
            # Print first 3 as examples
            if short_text_count <= 3:
                print(f"   - ID {row['subject_id']}: Text length = {text_len} chars")

        print(f"\nDiagnosis: {short_text_count} of {len(missing_ids)} missing records had text < 50 characters.")
        
        real_success_rate = len(success_ids) / total_gt
        adjusted_success_rate = len(success_ids) / (total_gt - short_text_count)
        
        print("\n" + "="*40)
        print("ACTUAL STATISTICS:")
        print(f"   - Raw rate (vs total CSV): {real_success_rate:.2%}")
        print(f"   - Adjusted rate (vs valid texts): {adjusted_success_rate:.2%}")
        print("="*40)

if __name__ == "__main__":
    audit_discrepancy()