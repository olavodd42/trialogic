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

def load_datasets(csv_path, experiments_dict):
    """Carrega GT e N experimentos, retornando dict de DataFrames mergeados."""
    logger.info(f"📂 Loading Ground Truth: {csv_path}")
    df_gt = pd.read_csv(csv_path)
    
    # Normaliza ID do GT
    id_col = 'hadm_id' if 'hadm_id' in df_gt.columns else 'stay_id'
    if id_col in df_gt.columns:
        df_gt[id_col] = pd.to_numeric(df_gt[id_col], errors='coerce')
    
    merged_results = {}
    
    for label, path in experiments_dict.items():
        if not os.path.exists(path):
            logger.warning(f"⚠️ File not found for {label}: {path}")
            continue
            
        logger.info(f"   🔹 Loading {label}: {path}")
        preds = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try: preds.append(json.loads(line))
                except: pass
        
        df_pred = pd.DataFrame(preds)
        if id_col in df_pred.columns:
            df_pred[id_col] = pd.to_numeric(df_pred[id_col], errors='coerce')
            # Merge
            merged = pd.merge(df_gt, df_pred, on=id_col, suffixes=('_gt', '_pred'))
            merged_results[label] = merged
        else:
            logger.warning(f"❌ ID Column {id_col} not found in {label}. Skipping.")
    
    return merged_results

# --- 1. INDEPENDENT ORACLE (GROUND TRUTH CALCULATOR) ---
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

    # Ensure that hadm_id is of same type
    if 'hadm_id' in df_gt.columns:
        df_gt['hadm_id'] = pd.to_numeric(df_gt['hadm_id'], errors='coerce')

    # Load predictions
    predictions = []
    with open(jsonl_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
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
        y_true, y_pred = [],[]
        
        outliers, valid_count = 0, 0
        for idx, row in df.iterrows():
            gt_val = row.get(gt_col)

            pred_val = None
            try:
                # Baseline case
                if 'extracted_vitals' in row and isinstance(row['extracted_vitals'], dict):
                    pred_val = row['extracted_vitals'].get(pred_key)
                # Trialogic case
                elif 'extracted_data' in row:
                    ext = row['extracted_data']
                    if isinstance(ext, str): ext = json.loads(ext)
                    pred_val = ext.get('vitals', {}).get(pred_key)
            except: 
                pass
                
            # Evaluation Logic
            if pd.notna(gt_val) and pred_val is not None:
                try:
                    gt_f, pred_f = float(gt_val), float(pred_val)
                
                    # --- Santizing Filter ---
                    if not (min_val <= pred_f <= max_val): 
                        outliers += 1
                        continue 

                    # Temperature Normalization (F to C)
                    if label == 'Temperature':
                        if gt_f > 50 and pred_f < 50: gt_f = (gt_f - 32) * 5/9
                        elif pred_f > 50 and gt_f < 50: pred_f = (pred_f - 32) * 5/9

                    if abs(gt_f - pred_f) > 100:
                        logger.warning(f"🚨 HUGE ERROR DETECTED in {label} (Idx {idx}): GT={gt_f} vs Pred={pred_f}")

                    y_true.append(gt_f)
                    y_pred.append(pred_f)
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

def evaluate_risk(df, name):
    logger.info("\n=== 2. REASONING EVALUATION (MATHEMATICIAN AGENT) ===")
    
    score_types = [
        ("NEWS2", oracle_news2, NEWS_PATTERN),
        ("MEWS", oracle_mews, MEWS_PATTERN)
    ]

    
    for label, oracle_func, regex_pattern in score_types:
        y_true, y_pred = [],[]

        for _, row in df.iterrows():
            # 1. Ground Truth
            gt = oracle_func(row)
            
            # 2. Prediction
            pred = parse_agent_score(row.get('risk_score', ''), regex_pattern)

            if pd.notna(gt) and pred is not None:
                y_true.append(int(gt))
                y_pred.append(int(pred))
                
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            logger.info(f"--- {label} ---")
            logger.info(f"Exact Score Accuracy: {acc:.1%}")
            logger.info(f"MAE (Deviation):      {mae:.2f}")
            
            # Confusion Metrics (Risk Categories)
            def cat(s): return 'Low' if s<=4 else ('Medium' if s<=6 else 'High')
            if label == 'MEWS':
                 def cat(s): return 'Low' if s<=2 else ('Medium' if s<=4 else 'High')
            

            c_true = [cat(s) for s in y_true]
            c_pred = [cat(s) for s in y_pred]

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
                plt.savefig(f'confusion_matrix_{label}_{name}.png')
                logger.info(f"Confusion matrix saved as 'confusion_matrix_{label}.png'")
                plt.close()
            except Exception as e:
                logger.warning(f"Could not save plot: {e}")

            logger.info("-" * 30)
            
        else:
            logger.warning(f"--- {label} ---")
            logger.warning("It wasn't possible to compare scores (insufficient data).")

def compute_metrics(df, system_name):
    results = {}
    
    # 3.1 Vitals Extraction
    vitals_config = [
        ('triage_heartrate', 'heartrate', 'MAE_HR', 0, 300),
        ('triage_sbp', 'sbp', 'MAE_SBP', 0, 300),
        ('triage_resprate', 'resprate', 'MAE_RR', 0, 100),
        ('triage_o2sat', 'o2sat', 'MAE_O2', 0, 100),
        ('triage_temperature', 'temperature', 'MAE_Temp', 0, 120)
    ]

    logger.debug("Calculating metrics...")
    for gt_col, pred_key, metric_name, min_v, max_v in vitals_config:
        y_true, y_pred = [], []
        for _, row in df.iterrows():
            gt_val = row.get(gt_col)
            pred_val = None
            try:
                # Baseline vs TriaLogic Polymorfism
                if 'extracted_vitals' in row and isinstance(row['extracted_vitals'], dict):
                    pred_val = row['extracted_vitals'].get(pred_key)
                elif 'extracted_data' in row:
                    ext = row['extracted_data']
                    if isinstance(ext, str): ext = json.loads(ext)
                    pred_val = ext.get('vitals', {}).get(pred_key)
            except: pass

            if pd.notna(gt_val) and pred_val is not None:
                try:
                    gt_f, pred_f = float(gt_val), float(pred_val)
                    if not (min_v <= pred_f <= max_v): continue # Filter outliers
                    
                    # Temperature Normalization
                    if 'Temp' in metric_name:
                        if gt_f > 50 and pred_f < 50: gt_f = (gt_f - 32) * 5/9
                        elif pred_f > 50 and gt_f < 50: pred_f = (pred_f - 32) * 5/9
                        
                    y_true.append(gt_f)
                    y_pred.append(pred_f)
                except: pass
        
        if y_true:
            results[metric_name] = mean_absolute_error(y_true, y_pred)
        else:
            results[metric_name] = None

    # 3.2 Risk Scoring (NEWS2/MEWS)
    def parse_score(row):
        logger.debug("Parsing scores...")
        txt = str(row.get('risk_score', ''))
        n = NEWS_PATTERN.search(txt)
        m = MEWS_PATTERN.search(txt)
        return (int(n.group(1)) if n else None, int(m.group(1)) if m else None)

    news_true, news_pred = [], []
    
    for _, row in df.iterrows():
        gt_news = oracle_news2(row)
        p_news, _ = parse_score(row)
        
        if pd.notna(gt_news) and p_news is not None:
            news_true.append(gt_news)
            news_pred.append(p_news)
            
    if news_true:
        results['Acc_NEWS2'] = accuracy_score(news_true, news_pred)
        
        # Risk Category Accuracy
        def cat(s): return 'Low' if s<=4 else ('Medium' if s<=6 else 'High')
        c_true = [cat(x) for x in news_true]
        c_pred = [cat(x) for x in news_pred]
        results['Risk_Cat_Acc'] = accuracy_score(c_true, c_pred)
        
        # High Risk Recall (Critical for Sepsis)
        cm = confusion_matrix(c_true, c_pred, labels=['Low', 'Medium', 'High'])
        high_idx = 2
        if sum(cm[high_idx, :]) > 0:
            results['Recall_High'] = cm[high_idx, high_idx] / sum(cm[high_idx, :])
        else:
            results['Recall_High'] = 0.0
            
    # Success Rate (Valid JSONs)
    valid_json = df['extracted_vitals'].notna().sum() if 'extracted_vitals' in df.columns else df['extracted_data'].notna().sum()
    results['Valid_JSON_Rate'] = valid_json / len(df)

    return results

# --- MAIN ---

if __name__ == "__main__":
    # Use absolute paths or relative to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(BASE_DIR, "data/master_dataset.csv") 

    EXPERIMENTS = {
        "Baseline (Zero-Shot)": os.path.join(BASE_DIR, "results/baseline_results.jsonl"),
        "TriaLogic (Agents)":   os.path.join(BASE_DIR, "results/experiment_results_v1.jsonl")
    }

    
    if not os.path.exists(CSV_PATH):
        logger.error("Dataset not found!")
        exit()

    # 1. Load and merge
    dfs = load_datasets(CSV_PATH, EXPERIMENTS)

    # 2. Calculate metrics
    final_table = []
    for name, df in dfs.items():
        logger.info(f"⚙️  Processing metrics for {name}...")
        metrics = compute_metrics(df, name)
        metrics['System'] = name
        final_table.append(metrics)
    
    # 3. Display final table
    if final_table:
        result_df = pd.DataFrame(final_table)
        cols = ['System', 'Valid_JSON_Rate', 'Acc_NEWS2', 'Risk_Cat_Acc', 'Recall_High', 'MAE_HR', 'MAE_SBP', 'MAE_Temp']
        cols = [c for c in cols if c in result_df.columns]

        print("\n" + "="*60)
        print("🏆  FINAL BENCHMARK RESULTS (TABLE 1)  🏆")
        print("="*60)

        format_dict = {
            'Valid_JSON_Rate': '{:.1%}',
            'Acc_NEWS2': '{:.1%}',
            'Risk_Cat_Acc': '{:.1%}',
            'Recall_High': '{:.1%}',
            'MAE_HR': '{:.2f}', 
            'MAE_SBP': '{:.2f}', 
            'MAE_Temp': '{:.2f}'
        }

        print(result_df[cols].to_string(formatters={
            k: v.format for k, v in format_dict.items() if k in result_df[cols].columns
        }))
        print("="*60)
        print("Interpretation Guide:")
        print("- Valid_JSON_Rate: Robustness (Did it crash?)")
        print("- Acc_NEWS2: Reasoning capability (Math)")
        print("- MAE_*: Extraction precision (Lower is better)")
        print("="*60)

    else:
        logger.error("No valid experiments found.")