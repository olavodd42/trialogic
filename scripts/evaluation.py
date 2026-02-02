import pandas as pd
import json
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import logging
import warnings

# --- CONFIGURAÇÃO & LOGGING ---
# Tech Lead: Logs claros salvam vidas (e debugging na madrugada).
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Orientador: Ignoramos warnings de divisão por zero em casos de suporte vazio, 
# mas mantemos o rigor nos logs de erro.
warnings.filterwarnings('ignore', category=RuntimeWarning) 

# --- CONSTANTES & PADRÕES ---
NEWS_PATTERN = re.compile(r"SCORE TOTAL NEWS:\s*(\d+)")
MEWS_PATTERN = re.compile(r"SCORE TOTAL MEWS:\s*(\d+)")

# Mapeamento de colunas para garantir consistência entre CSV e Extração
METRIC_MAPPING = {
    'sbp': 'SBP',
    'dbp': 'DBP',
    'heartrate': 'HR',
    'resprate': 'RR',
    'o2sat': 'O2',
    'temperature_celsius': 'Temp',
    'mews': 'MEWS',
    'news': 'NEWS',
    'acvpu': 'AVPU' # Ajuste conforme necessário
}


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


# 3. --- METRICS ENGINE ---
def compute_metrics(df, system_name):
    results = {}

    # 3.1 Hallucination rate & Parsing success rate
    valid_json_count = 0
    hallucination_count = 0
    total_extractions = 0

    guardrails = {
        'heartrate': (0, 300),
        'sbp': (0, 300),
        'dbp': (0, 200),
        'resprate': (0, 100),
        'o2sat': (0, 100),
        'temperature': (0, 115)
    }
    
    # 3.2 Entity Extraction
    extraction_targets = [
        ('heartrate', 'heartrate', 'HR', 5.0),
        ('sbp', 'sbp', 'SBP', 5.0),
        ('resprate', 'resprate', 'RR', 2.0),
        ('o2sat', 'o2sat', 'O2', 2.0),
        ('temperature', 'temperature', 'Temp', 0.5)
    ]
    f1_scores = []
    precision_scores = []
    recall_scores = []

    logger.debug("Calculating metrics...")
    for gt_col, pred_key, label, tol in extraction_targets:
        tp, fn, fp = 0, 0, 0
        for _, row in df.iterrows():
            is_valid_json = False
            pred_val = None
            try:
                # Baseline vs TriaLogic Polymorfism
                if 'extracted_vitals' in row and isinstance(row['extracted_vitals'], dict):
                    pred_val = row['extracted_vitals'].get(pred_key)
                    is_valid_json = True
                elif 'extracted_data' in row:
                    ext = row['extracted_data']
                    if isinstance(ext, str): ext = json.loads(ext)
                    pred_val = ext.get('vitals', {}).get(pred_key)
                    is_valid_json = True
            except: pass

            if is_valid_json and pred_key == 'heartrate':
                valid_json_count += 1


            if pred_val is not None:
                try:
                    val_float = float(pred_val)
                    min_g, max_g = guardrails.get(pred_key, (0, 1000))
                    if not (min_g <= val_float <= max_g):
                        hallucination_count += 1
                    total_extractions += 1
                except: pass
        
            # F1 Calc Logic
            gt_val = row.get(gt_col)
            has_gt = pd.notna(gt_val)
            has_pred = pred_val is not None

            if has_gt and has_pred:
                try:
                    gt_f, pred_f = float(gt_val), float(pred_val)
                    if label == 'Temp':
                        if gt_f > 50 and pred_f < 50: gt_f = (gt_f - 32) * 5/9
                        elif pred_f > 50 and gt_f < 50: pred_f = (pred_f - 32) * 5/9
                    
                    if abs(gt_f - pred_f) <= tol:
                        tp += 1
                    else:
                        fp += 1
                except: fp += 1
            elif has_gt and not has_pred:
                fn += 1
            elif not has_gt and has_pred:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        results[f'Prec_{label}'] = precision
        results[f'Rec_{label}'] = recall
        results[f'F1_{label}'] = f1

    results['Macro_F1'] = np.mean(f1_scores) if f1_scores else 0
    results['Mean_Precision'] = np.mean(precision_scores) if precision_scores else 0
    results['Mean_Recall'] = np.mean(recall_scores) if recall_scores else 0
    results['Hallucination_Rate'] = hallucination_count / total_extractions if total_extractions > 0 else 0
    results['Parsing_Success_Rate'] = valid_json_count / len(df)

    # 3.2 Risk Scoring (NEWS2/MEWS)
    def parse_score(row):
        logger.debug("Parsing scores...")
        txt = str(row.get('risk_score', ''))
        n = NEWS_PATTERN.search(txt)
        m = MEWS_PATTERN.search(txt)
        return (int(n.group(1)) if n else None, int(m.group(1)) if m else None)

    # 3.3 Clinical Score Accuracy (SEM - Strict Exact Match)
    # news_true, news_pred = [], []
    sem_news_matches = 0
    sem_mews_matches = 0
    valid_news_comparisons = 0
    valid_mews_comparisons = 0

    for _, row in df.iterrows():
        gt_news = row["news_gt"]
        gt_mews = row["mews_gt"]
        p_news, p_mews = parse_score(row)
        
        if pd.notna(gt_news) and p_news is not None:
            valid_news_comparisons += 1
            if int(gt_news) == int(p_news):
                sem_news_matches += 1
        
        if pd.notna(gt_mews) and p_mews is not None:
            valid_mews_comparisons += 1
            if int(gt_mews) == int(p_mews):
                sem_mews_matches += 1

    results['SEM_NEWS2'] = sem_news_matches / valid_news_comparisons if valid_news_comparisons > 0 else 0
    results['SEM_MEWS'] = sem_mews_matches / valid_mews_comparisons if valid_mews_comparisons > 0 else 0
            

    return results
# --- MAIN ---

if __name__ == "__main__":
    # Use absolute paths or relative to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(BASE_DIR, "data/master_dataset.csv") 

    EXPERIMENTS = {
        "Baseline (Zero-Shot)": os.path.join(BASE_DIR, "results/baseline_results.jsonl"),
        "Baseline (One-Shot)": os.path.join(BASE_DIR, "results/oneshot_baseline_results.jsonl"),
        "TriaLogic (No RAG)": os.path.join(BASE_DIR, "results/norag_experiment_results_v1.jsonl"),
        "TriaLogic (Agents)": os.path.join(BASE_DIR, "results/experiment_results_v1.jsonl")
    }

    
    if not os.path.exists(CSV_PATH):
        logger.error("Dataset not found!")
        exit()

    # 1. Load and merge
    dfs = load_datasets(CSV_PATH, EXPERIMENTS)

    # 2. Calculate metrics, garantindo todos os experimentos
    final_table = []
    metric_keys = [
        'Mean_Precision', 'Mean_Recall', 'Macro_F1', 'Parsing_Success_Rate', 'Hallucination_Rate', 'SEM_NEWS2', 'SEM_MEWS',
        'Prec_HR', 'Rec_HR', 'F1_HR', 'Prec_SBP', 'Rec_SBP', 'F1_SBP', 'Prec_RR', 'Rec_RR', 'F1_RR', 'Prec_O2', 'Rec_O2', 'F1_O2', 'Prec_Temp', 'Rec_Temp', 'F1_Temp'
    ]
    for name in EXPERIMENTS.keys():
        df = dfs.get(name, None)
        try:
            if df is not None and not df.empty:
                logger.info(f"⚙️  Processing metrics for {name}...")
                metrics = compute_metrics(df, name)
                metrics['System'] = name
            else:
                raise ValueError("DataFrame vazio ou não encontrado")
        except Exception as e:
            logger.warning(f"⚠️  No data for {name} (motivo: {e}). Filling with NaN.")
            metrics = dict(System=name)
            for k in metric_keys:
                metrics[k] = float('nan')
        final_table.append(metrics)

    # 3. Display final table
    if final_table:
        result_df = pd.DataFrame(final_table)
        cols_summary = ['System', 'Mean_Precision', 'Mean_Recall', 'Macro_F1', 'Parsing_Success_Rate', 'Hallucination_Rate', 'SEM_NEWS2', 'SEM_MEWS']
        # Garante ordem e presença de todos os sistemas
        result_df = result_df.set_index('System').reindex(list(EXPERIMENTS.keys())).reset_index()
        print("\n" + "="*80)
        print("🏆  TABLE 1: SYSTEM OVERVIEW  🏆")
        print("="*80)
        format_dict = {
            'Mean_Precision': '{:.3f}', 
            'Mean_Recall': '{:.3f}', 
            'Macro_F1': '{:.3f}', 
            'SEM_NEWS2': '{:.4f}', 
            'SEM_MEWS': '{:.4f}', 
            'Hallucination_Rate': '{:.2%}',
            'Parsing_Success_Rate': '{:.2%}'
        }
        print(result_df[cols_summary].to_string(formatters={
            k: v.format for k, v in format_dict.items()
        }))
        print("="*80)
        # --- TABLE 2: DETAILED EXTRACTION ---
        print("\n" + "="*80)
        print("🔬  TABLE 2: DETAILED EXTRACTION PERFORMANCE (Precision / Recall / F1)  🔬")
        print("="*80)
        
        # Dynamically find per-target cols
        targets = ['HR', 'SBP', 'RR', 'O2', 'Temp']
        detailed_cols = ['System']
        for t in targets:
            detailed_cols.extend([f'Prec_{t}', f'Rec_{t}', f'F1_{t}'])
        
        # Helper to format compact
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        # Transpose or format nicely? Let's just print filtered columns with 2 decimal places
        subset = result_df[detailed_cols]
        print(subset.round(2).to_string(index=False))
        print("="*80)
    else:
        logger.error("No valid experiments found.")