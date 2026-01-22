import pandas as pd
import json
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, classification_report
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

NEWS_PATTERN = re.compile(r"SCORE TOTAL NEWS:\s*(\d+)")
MEWS_PATTERN = re.compile(r"SCORE TOTAL MEWS:\s*(\d+)")

# --- 1. INDEPENDENT ORACLE (GROUND TRUTH CALCULATOR) ---
def infer_avpu_from_csv(row):
    cc = str(row.get('chiefcomplaint', '')).lower()
    triggers = ['confusion', 'confused', 'disoriented', 'altered mental', 'lethargic', 'unresponsive', 'obtunded']
    
    if any(t in cc for t in triggers):
        return 3
    return 0

def oracle_news2(row):
    score = 0
    
    if pd.isna(row.get('triage_sbp')) or pd.isna(row.get('triage_heartrate')):
        return np.nan

    # HR
    hr = row['triage_heartrate']
    if hr <= 40 or hr >= 131: score += 3
    elif 111 <= hr <= 130: score += 2
    elif 41 <= hr <= 50: score += 1
    
    # SBP
    sbp = row['triage_sbp']
    if sbp <= 90: score += 3
    elif 91 <= sbp <= 100: score += 2
    elif 101 <= sbp <= 110: score += 1
    
    # Resp Rate
    rr = row.get('triage_resprate', 18) # Default 18 se faltar
    if rr <= 8 or rr >= 25: score += 3
    elif 21 <= rr <= 24: score += 2
    elif 9 <= rr <= 11: score += 1
    
    # Temp
    temp = row.get('triage_temperature')
    if pd.notna(temp):
        if temp > 50: temp = (temp - 32) * 5/9 # F to C
        
        if temp <= 35.0: score += 3
        elif temp >= 39.1: score += 2
        elif 35.1 <= temp <= 36.0 or 38.1 <= temp <= 39.0: score += 1

    # O2 Sat
    o2 = row.get('triage_o2sat')
    if pd.notna(o2):
        if o2 <= 91: score += 3
        elif 92 <= o2 <= 93: score += 2
        elif 94 <= o2 <= 95: score += 1

    # Supplemental Oxygen (Assume False as lacking data, matches extracting 'supplemental_oxygen' 0 in most cases if unknown)
    score += 0 
    
    # Consciousness (Using Acuity as Proxy because AVPU is missing)
    # ESI 1/2 -> Likely altered mental status
    acuity = row.get('triage_acuity')
    if pd.notna(acuity):
        if acuity <= 2: 
            score += 3
        elif (rr < 8) or (rr > 30) or (sbp < 90) or (o2 < 90) or (hr > 130) or (hr < 40):
            score += 3
    
    return score

def oracle_mews(row):
    score = 0
    
    if pd.isna(row.get('triage_sbp')) or pd.isna(row.get('triage_heartrate')):
        return np.nan

    # 1. SBP
    sbp = row['triage_sbp']
    if sbp <= 70: score += 3
    elif 71 <= sbp <= 80: score += 2
    elif 81 <= sbp <= 100: score += 1
    elif sbp >= 200: score += 2 # MEWS penaliza hipertensão grave

    # 2. HR
    hr = row['triage_heartrate']
    if hr <= 40: score += 2
    elif 41 <= hr <= 50: score += 1
    elif 101 <= hr <= 110: score += 1
    elif 111 <= hr <= 129: score += 2
    elif hr >= 130: score += 3

    # 3. RR
    rr = row.get('triage_resprate', 18)
    if rr <= 8: score += 2
    elif 15 <= rr <= 20: score += 1
    elif 21 <= rr <= 29: score += 2
    elif rr >= 30: score += 3

    # 4. Temp (C)
    temp = row.get('triage_temperature')
    if pd.notna(temp):
        if temp > 50: temp = (temp - 32) * 5/9
        if temp <= 35.0: score += 2
        elif temp >= 38.5: score += 2

    # 5. AVPU (Default 0, try Acuity)
    acuity = row.get('triage_acuity')
    if pd.notna(acuity):
        if acuity <= 1: score += 3 # Unresponsive
        elif acuity == 2: score += 2 # Pain/Voice? Heuristic.
        elif (rr < 8) or (rr > 30) or (sbp < 90) or (hr > 130) or (hr < 40):
            score += 2
        # else 0 (Alert)
    
    return score

# --- 2. ROBUST LOADERS ---

def load_data(csv_path, jsonl_results_path):
    logger.info(f"Loading datasets from:\nCSV: {csv_path}\nJSONL: {jsonl_results_path}")
    
    # Load Ground Truth
    df_gt = pd.read_csv(csv_path)
    # Ensure hadm_id is same type
    if 'hadm_id' in df_gt.columns:
        df_gt['hadm_id'] = pd.to_numeric(df_gt['hadm_id'], errors='coerce')

    # Load predictions
    predictions = []
    with open(jsonl_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    df_pred = pd.DataFrame(predictions)
    if 'hadm_id' in df_pred.columns:
        df_pred['hadm_id'] = pd.to_numeric(df_pred['hadm_id'], errors='coerce')
    
    # Merge with hadm_id
    if 'hadm_id' in df_gt.columns and 'hadm_id' in df_pred.columns:
        logger.info(f"Merging data with hadm_id... (GT: {len(df_gt)}, Pred: {len(df_pred)})")
        merged = pd.merge(df_gt, df_pred, on='hadm_id', suffixes=('_gt', '_pred'))
        logger.info(f"Merged Result: {len(merged)} records.")
    else:
        logger.warning("⚠️ Merge IDs not found. Assuming sequencial order (Risk of Misalignment).")
        min_len = min(len(df_gt), len(df_pred))
        merged = pd.concat([df_gt.iloc[:min_len].reset_index(drop=True), 
                           df_pred.iloc[:min_len].reset_index(drop=True)], axis=1)
    
    return merged

# --- 3. EXTRACTION EVALUATION (NER) ---

def evaluate_vitals(df):
    logger.info("\n=== 1. EXTRACTION EVALUATION (SCRIBE AGENT) ===")
    
    metrics = [
        ('triage_heartrate', 'heartrate', 'Heart Rate', 0, 300), # Range válido
        ('triage_sbp', 'sbp', 'SBP', 0, 300),
        ('triage_dbp', 'dbp', 'DBP', 0, 200),
        ('triage_resprate', 'resprate', 'Resp Rate', 0, 100),
        ('triage_o2sat', 'o2sat', 'O2 Sat', 0, 100),
        ('triage_temperature', 'temperature', 'Temperature', 0, 120)
    ]

    
    report = []
    
    for gt_col, pred_key, label, min_val, max_val in metrics:        
        y_true = []
        y_pred = []
        
        outliers = 0
        valid_count = 0
        # missing_pred_count = 0
        for idx, row in df.iterrows():
            gt_val = row.get(gt_col)

            pred_val = None
            try:
                # Try 'extracted_vitals' first (Direct dict)
                vitals = {}
                if 'extracted_vitals' in row: 
                    raw_vitals = row['extracted_vitals']
                    if pd.notna(raw_vitals):
                        vitals = raw_vitals if isinstance(raw_vitals, dict) else json.loads(raw_vitals)
                
                # Fallback to 'extracted_data' (Nested dict)
                elif 'extracted_data' in row:
                    raw_ext = row['extracted_data']
                    if pd.notna(raw_ext):
                        ext_data = raw_ext if isinstance(raw_ext, dict) else json.loads(raw_ext)
                        vitals = ext_data.get('vitals', {})
                
                # pred_key already defined in loop var
                pred_val = vitals.get(pred_key)
                
            except Exception:
                # logger.error("Extracted data couldn't be loaded")
                pass

            # Evaluation Logic
            if pd.notna(gt_val) and pred_val is not None:
                try: 
                    gt_float = float(gt_val)
                    pred_float = float(pred_val)
                    
                    # --- Santizing Filter ---
                    if not (min_val <= pred_float <= max_val):
                        outliers += 1
                        continue
                    if not (min_val <= pred_float <= max_val):
                        outliers += 1
                        continue 

                    # Temperature Normalization (F to C)
                    if label == 'Temperature':
                        if gt_float > 50 and pred_float < 50:
                            gt_float = (gt_float - 32) * 5/9
                        elif pred_float > 50 and gt_float < 50:
                            pred_float = (pred_float - 32) * 5/9

                    if abs(gt_float - pred_float) > 100:
                        logger.warning(f"🚨 HUGE ERROR DETECTED in {label} (Idx {idx}): GT={gt_float} vs Pred={pred_float}")

                    y_true.append(gt_float)
                    y_pred.append(pred_float)
                    valid_count += 1
                except ValueError:
                    pass
        
        # Metrics Computation
        if y_true:
            mae = mean_absolute_error(y_true, y_pred)
            tol = 0.5 if label == 'Temperature' else 5.0
            correct = sum([1 for t, p in zip(y_true, y_pred) if abs(t - p) <= tol])
            acc = correct / len(y_true)
            
            logger.info(f"{label:<15} | MAE: {mae:.2f} | Acc (+-{tol}): {acc:.1%} | N: {valid_count} | Outliers Removidos: {outliers}")
            
            report.append({
                "Metric": label, 
                "MAE": mae, 
                "Accuracy": acc
            })
        else:
            logger.warning(f"{label:<15} | Insufficient Data")

    return pd.DataFrame(report)

def parse_agent_score(risk_score_str, pattern):
    if not isinstance(risk_score_str, str): return None
    match = pattern.search(risk_score_str)
    if match:
        return int(match.group(1))
    return None

# --- 4. REASONING EVALUATION (CALCULATOR) ---

def evaluate_risk(df):
    logger.info("\n=== 2. REASONING EVALUATION (MATHEMATICIAN AGENT) ===")
    
    score_types = [
        ("NEWS2", oracle_news2, NEWS_PATTERN),
        ("MEWS", oracle_mews, MEWS_PATTERN)
    ]

    
    for label, oracle_func, regex_pattern in score_types:
        y_true = []
        y_pred = []
        for _, row in df.iterrows():
            # 1. Ground Truth
            gt = oracle_func(row)
            
            # 2. Prediction
            raw_text = row.get('risk_score', '')
            pred = parse_agent_score(raw_text, regex_pattern)

            if pd.notna(gt) and pred is not None:
                y_true.append(int(gt))
                y_pred.append(int(pred))
                
        if y_true:
            labels = ['Low', 'Medium', 'High']
            acc = accuracy_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            logger.info(f"--- {label} ---")
            logger.info(f"Exact Score Accuracy: {acc:.1%}")
            logger.info(f"MAE (Deviation):      {mae:.2f}")
            
            # Confusion Metrics (Risk Categories)
            def cat_news(s):
                if s <= 4: return 'Low'
                elif s <= 6: return 'Medium'
                else: return 'High'

            def cat_mews(s):
                # Using 0-2 (Low), 3-4 (Med), 5+ (High) is more standard for MEWS
                if s <= 2: return 'Low'
                if s <= 4: return 'Medium'
                return 'High'
            
            # Use specific categorizer or generic
            if label == 'NEWS2': 
                cat_func = cat_news
            else:
                cat_func = cat_mews

            c_true = [cat_func(s) for s in y_true]
            c_pred = [cat_func(s) for s in y_pred]

            cat_acc = accuracy_score(c_true, c_pred)
            logger.info(f"Risk Category Accuracy: {cat_acc:.1%}")

            labels = ['Low', 'Medium', 'High']
            cm = confusion_matrix(c_true, c_pred, labels=labels)
            
            high_idx = 2 
            recall_high = cm[high_idx, high_idx] / sum(cm[high_idx, :]) if sum(cm[high_idx, :]) > 0 else 0
            logger.info(f"Recall High Risk (Critical): {recall_high:.1%}")

            cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
            logger.info(f"\nConfusion Matrix ({label}):")
            logger.info("\n" + str(cm_df))
            
            # Save plot
            try:
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
                plt.title(f'{label} Risk Stratification')
                plt.ylabel('Ground Truth (MIMIC Data)')
                plt.xlabel('TriaLogic Prediction')
                plt.tight_layout()
                plt.savefig(f'confusion_matrix_{label}.png')
                logger.info(f"Confusion matrix saved as 'confusion_matrix_{label}.png'")
                plt.close()
            except Exception as e:
                logger.warning(f"Could not save plot: {e}")

            logger.info("-" * 30)
            
        else:
            logger.warning(f"--- {label} ---")
            logger.warning("It wasn't possible to compare scores (insufficient data).")

# --- MAIN ---

if __name__ == "__main__":
    # Use absolute paths or relative to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(BASE_DIR, "data/master_dataset.csv") 
    JSONL_PATH = os.path.join(BASE_DIR, "results/experiment_results_v1.jsonl")
    
    if not os.path.exists(JSONL_PATH):
        # Fallback to v2 if v1 doesn't exist
        JSONL_PATH = os.path.join(BASE_DIR, "results/experiment_results_v2.jsonl")

    try:
        if os.path.exists(CSV_PATH) and os.path.exists(JSONL_PATH):
            df_merged = load_data(CSV_PATH, JSONL_PATH)
            
            # Avaliação Faseada
            evaluate_vitals(df_merged)
            evaluate_risk(df_merged)
            
            logger.info("\n✅ Finished evaluation succesfully.")
        else:
            logger.error(f"Files not found. \nCSV: {CSV_PATH}\nJSONL: {JSONL_PATH}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Verify the paths.")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
