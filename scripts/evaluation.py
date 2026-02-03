import pandas as pd
import json
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
import logging
import warnings

# --- CONFIGURAÇÃO & LOGGING ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning) 

# --- CONSTANTES DE TOLERÂNCIA CLÍNICA (Para F1/Recall/Precision) ---
# Tech Lead: Margens de erro aceitáveis para considerar um "Acerto" (True Positive)
CLINICAL_TOLERANCES = {
    'sbp': 10.0,
    'dbp': 10.0,
    'heartrate': 5.0,
    'resprate': 2.0,
    'o2sat': 2.0,
    'temperature_celsius': 0.5,
}


# --- REGEX CONSTANTS ---
NEWS_PATTERN = re.compile(r"SCORE TOTAL NEWS:?\s*(\d+)", re.IGNORECASE)
MEWS_PATTERN = re.compile(r"SCORE TOTAL MEWS:?\s*(\d+)", re.IGNORECASE)

def normalize_value(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, str):
        v = v.strip().lower().replace('%', '').replace(',', '.')
        try:
            return float(v)
        except ValueError:
            return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan

def get_news_risk(score):
    if pd.isna(score): return None
    s = int(score)
    if s <= 4: return 'Low'
    if s <= 6: return 'Medium'
    return 'High'

def get_mews_risk(score):
    if pd.isna(score): return None
    s = int(score)
    if s <= 1: return 'Low'
    if s <= 3: return 'Medium'
    return 'High'

def extract_score_from_text(text, pattern):
    if not isinstance(text, str): return np.nan
    match = pattern.search(text)
    return int(match.group(1)) if match else np.nan

def load_json_or_jsonl(path):
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[': 
                data = json.load(f)
            else:
                for line in f:
                    if line.strip(): data.append(json.loads(line))
        return data
    except Exception as e:
        logger.error(f"❌ Error reading {path}: {e}")
        return []

def normalize_prediction(item):
    hadm_id = item.get('hadm_id') or item.get('stay_id')
    vitals = item.get('extracted_vitals') or {}


    risk_text = item.get('risk_score', '')


    return {
        'hadm_id': int(hadm_id) if hadm_id is not None else 0,
        'Pred_SBP': normalize_value(vitals.get('sbp')),
        'Pred_DBP': normalize_value(vitals.get('dbp')),
        'Pred_HR': normalize_value(vitals.get('heartrate')),
        'Pred_RR': normalize_value(vitals.get('resprate')),
        'Pred_O2': normalize_value(vitals.get('o2sat')),
        'Pred_Temp': normalize_value(vitals.get('temperature')),
        'Pred_NEWS': extract_score_from_text(risk_text, NEWS_PATTERN),
        'Pred_MEWS': extract_score_from_text(risk_text, MEWS_PATTERN),
    }

def load_and_sanitize_data(csv_path, experiments_dict):
    logger.info(f"📂 Loading GT: {csv_path}")
    try:
        df_gt = pd.read_csv(csv_path, na_values=['N/A', 'n/a', 'NA', 'null', 'None', '', 'BS'])
    except Exception as e:
        logger.error(f"❌ Error GT: {e}")
        return None

    id_col = 'hadm_id' if 'hadm_id' in df_gt.columns else 'stay_id'
    if id_col not in df_gt.columns: return None
    
    df_gt[id_col] = pd.to_numeric(df_gt[id_col], errors='coerce').fillna(0).astype(int)
    df_gt = df_gt.set_index(id_col)

    numeric_cols = ['sbp', 'dbp', 'heartrate', 'resprate', 'o2sat', 'temperature_celsius', 'mews', 'news']
    for col in numeric_cols:
        if col in df_gt.columns:
           df_gt[col] = df_gt[col].apply(normalize_value)

    merged_results = {}
    for label, path in experiments_dict.items():
        if not os.path.exists(path): continue
        
        raw_data = load_json_or_jsonl(path)
        clean_preds = [normalize_prediction(item) for item in raw_data if item]
        if not clean_preds: continue

        df_pred = pd.DataFrame(clean_preds)
        df_pred['hadm_id'] = pd.to_numeric(df_pred['hadm_id'], errors='coerce')
        df_pred = df_pred.dropna(subset=['hadm_id']).set_index(df_pred['hadm_id'].astype(int))
        
        merged_results[label] = df_gt.join(df_pred, how='inner')
        
    return merged_results

def calculate_clinical_metrics(merged_results):
    rows = []
    vitals_targets = [
        ('sbp', 'Pred_SBP', 'SBP', CLINICAL_TOLERANCES['sbp']),
        ('dbp', 'Pred_DBP', 'DBP', CLINICAL_TOLERANCES['dbp']),
        ('heartrate', 'Pred_HR', 'HR', CLINICAL_TOLERANCES['heartrate']),
        ('resprate', 'Pred_RR', 'RR', CLINICAL_TOLERANCES['resprate']),
        ('o2sat', 'Pred_O2', 'O2', CLINICAL_TOLERANCES['o2sat']),
        ('temperature_celsius', 'Pred_Temp', 'Temp', CLINICAL_TOLERANCES['temperature_celsius']),
    ]
    score_targets = [
        ('mews', 'Pred_MEWS', 'MEWS', get_mews_risk),
        ('news', 'Pred_NEWS', 'NEWS', get_news_risk),
    ]
    for system, df in merged_results.items():
        # --- VITAIS ---
        for gt_col, pred_col, label, tol in vitals_targets:
            if gt_col not in df.columns:
                continue
            mask = df[gt_col].notna() & df[pred_col].notna()
            if mask.sum() == 0:
                continue
            diff = (df.loc[mask, gt_col] - df.loc[mask, pred_col]).abs()
            within_tol = diff <= tol
            TP = within_tol.sum()
            FN = (~within_tol).sum()
            # Hallucination: predição fora do range fisiológico (exemplo simples: valor negativo ou muito alto)
            # Aqui, consideramos valores negativos ou acima de 300 como "alucinação" para sinais vitais
            pred_values = df.loc[mask, pred_col]
            halluc_mask = (pred_values < 0) | (pred_values > 300)
            halluc_rate = halluc_mask.mean() if len(pred_values) > 0 else np.nan
            # Precision e Recall: para este contexto, consideramos:
            # - TP: dentro da tolerância
            # - FP: previu, mas fora da tolerância
            # - FN: não previu (NaN), mas há GT (já filtrado pelo mask)
            FP = FN  # pois só há positivo (previu) e negativo (não previu ou fora da tolerância)
            precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            clin_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
            rows.append({
                'System': system,
                'Variable': label,
                'Metric_Type': 'Clin_F1',
                'Clinical_Score': clin_f1,
                'Precision': precision,
                'Recall': recall,
                'Strict_Match': (diff == 0).mean(),
                'MAE': diff.mean(),
                'Halluc_Rate': halluc_rate,
                'Samples': mask.sum(),
            })
        # --- SCORES (NEWS / MEWS) ---
        for gt_col, pred_col, label, risk_fn in score_targets:
            if gt_col not in df.columns:
                continue
            mask = df[gt_col].notna() & df[pred_col].notna()
            if mask.sum() == 0:
                continue
            gt_risk = df.loc[mask, gt_col].apply(risk_fn)
            pred_risk = df.loc[mask, pred_col].apply(risk_fn)
            valid = gt_risk.notna() & pred_risk.notna()
            if valid.sum() == 0:
                continue
            acc = (gt_risk[valid] == pred_risk[valid]).mean()
            # Hallucination: score predito fora do range esperado (ex: <0 ou >20)
            pred_score = df.loc[mask, pred_col]
            halluc_mask = (pred_score < 0) | (pred_score > 20)
            halluc_rate = halluc_mask.mean() if len(pred_score) > 0 else np.nan
            # Precision/Recall para classificação de risco (Low/Medium/High)
            # Calcula para cada classe e faz média macro
            from sklearn.metrics import precision_score, recall_score, f1_score
            y_true = gt_risk[valid]
            y_pred = pred_risk[valid]
            labels = ['Low', 'Medium', 'High']
            try:
                precision = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
            except Exception:
                precision = recall = f1 = np.nan
            rows.append({
                'System': system,
                'Variable': label,
                'Metric_Type': 'Risk_Acc',
                'Clinical_Score': acc,
                'Precision': precision,
                'Recall': recall,
                'Strict_Match': (df.loc[mask, gt_col] == df.loc[mask, pred_col]).mean(),
                'MAE': (df.loc[mask, gt_col] - df.loc[mask, pred_col]).abs().mean(),
                'Halluc_Rate': halluc_rate,
                'Samples': valid.sum(),
            })
    return pd.DataFrame(rows)

def print_final_report(df_metrics):
    if df_metrics.empty:
        print("⚠️ No data.")
        return

    print("\n" + "=" * 110)
    print("🏆 TCC EVALUATION: VITAL SIGNS vs CLINICAL SCORES 🏆")
    print("=" * 110)

    for sys in df_metrics['System'].unique():
        print(f"\n👉 SYSTEM: {sys}")
        subset = df_metrics[df_metrics['System'] == sys]
        # Exibe todas as métricas relevantes
        display_cols = [
            'Variable', 'Metric_Type', 'Clinical_Score', 'Precision', 'Recall',
            'Strict_Match', 'MAE', 'Halluc_Rate', 'Samples'
        ]
        # Ajusta para mostrar só colunas presentes
        display_cols = [c for c in display_cols if c in subset.columns]
        print(subset[display_cols].round(4).to_string(index=False))

# --- MAIN ---

if __name__ == "__main__":
    # Use absolute paths or relative to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GT_PATH = os.path.join(BASE_DIR, "data/ground_truth.csv")

    EXPERIMENTS = {
        "Baseline (Zero-Shot)": os.path.join(BASE_DIR, "results/baseline_results.jsonl"),
        "Baseline (One-Shot)": os.path.join(BASE_DIR, "results/oneshot_baseline_results.jsonl"),
        "No RAG": os.path.join(BASE_DIR, "results/norag_experiment_results_v1.jsonl"),
        "TriaLogic (Agents)": os.path.join(BASE_DIR, "results/experiment_results_v1.jsonl")
    }

    if os.path.exists(GT_PATH):
        print(f"🚀 Starting evaluation on: {GT_PATH}")
        data = load_and_sanitize_data(GT_PATH, EXPERIMENTS)
        
        if data:
            metrics_df = calculate_clinical_metrics(data)
            print_final_report(metrics_df)
            
            # Exportar para CSV
            output_csv = "results/tcc_final_metrics.csv"
            metrics_df.to_csv(output_csv, index=False)
            print(f"\n💾 Results saved to '{output_csv}'")
    else:
        logger.error(f"❌ Ground Truth file not found: {GT_PATH}")