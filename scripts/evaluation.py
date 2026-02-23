# scripts/evaluation_improved.py
"""
Evaluation improved for TriaLogic
- Robust hallucination detection
- rag_context_used capture & aggregation
- Single consolidated terminal table (MAE + Halluc_Rate)
- Exports: CSV, LaTeX, HTML, Markdown, per-system HTML
Usage: python -m scripts.evaluation_improved
"""

import os
import re
import json
import logging
import warnings
from typing import Dict, Any, Tuple, List, Optional
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ----------------- Clinical config -----------------
PHYSIO_RANGES = {
    'sbp': (30, 300),
    'dbp': (20, 200),
    'heartrate': (20, 300),
    'resprate': (5, 60),
    'o2sat': (50, 100),
    'temperature_celsius': (30.0, 43.0),
}

SCORE_EXTREME_DIFF_THRESHOLD = 5
RELATIVE_ERROR_THRESHOLD = 2.0
CONST_VALUE_SUSPECT_STD = 0.01

NEWS_PATTERNS = [
    re.compile(r"SCORE TOTAL NEWS:?\s*(\d+)", re.IGNORECASE),
    re.compile(r"NEWS[^0-9]{0,10}score\s*[:=]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"NEWS\s*score\s*[:=]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"NEWS:\s*score=(\d+)", re.IGNORECASE),           # probabilistic format
    re.compile(r"NEWS\s*[:\-]?\s*(\d+)", re.IGNORECASE),
]

MEWS_PATTERNS = [
    re.compile(r"SCORE TOTAL MEWS:?\s*(\d+)", re.IGNORECASE),
    re.compile(r"MEWS[^0-9]{0,10}score\s*[:=]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"MEWS\s*score\s*[:=]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"MEWS:\s*score=(\d+)", re.IGNORECASE),           # probabilistic format
    re.compile(r"MEWS\s*[:\-]?\s*(\d+)", re.IGNORECASE),
]

# Alternate candidate names for GT columns (makes matching robust)
GT_CANDIDATES = {
    'sbp': ['sbp', 'systolic', 'systolic_bp', 'sbp_mmhg'],
    'dbp': ['dbp', 'diastolic', 'diastolic_bp', 'dbp_mmhg'],
    'heartrate': ['heartrate', 'hr', 'pulse'],
    'resprate': ['resprate', 'rr', 'resp_rate', 'respiratory_rate'],
    'o2sat': ['o2sat', 'spo2', 'o2_sat', 'oxygen_saturation'],
    'temperature_celsius': ['temperature_celsius', 'temperature', 'temp', 'temperature_c'],
    'mews': ['mews', 'mews_score'],
    'news': ['news', 'news_score']
}


# ----------------- Utils -----------------
def normalize_value(v):
    if v is None: return np.nan
    if isinstance(v, float) and np.isnan(v): return np.nan
    # note: pandas may give us built-in Python floats (not np.floating), so include float explicitly
    if isinstance(v, (int, float, np.integer, np.floating)):
        try: return float(v)
        except: return np.nan
    if isinstance(v, str):
        s = v.strip().lower().replace('%','').replace(',','.')
        if s in ('', 'na', 'none', 'null'):
            return np.nan
        try:
            return float(s)
        except:
            return np.nan
    return np.nan

def extract_score_from_text(text, patterns: List[re.Pattern]):
    if not isinstance(text, str): return np.nan
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except:
                continue
    fallback = re.search(r"SCORE TOTAL\s*(?:NEWS|MEWS)[:\s]*([0-9]{1,2})", text, re.IGNORECASE)
    if fallback:
        return int(fallback.group(1))
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

def find_first_present(df_cols: List[str], candidates: List[str]) -> str:
    """
    Returns first candidate that exists in df_cols, else empty string.
    """
    cols_lower = {c.lower(): c for c in df_cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return ""


# ----------------- I/O JSONL -----------------
def load_json_or_jsonl(path: str) -> List[Dict[str,Any]]:
    results = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            first = f.read(1)
            f.seek(0)
            if first == '[':
                results = json.load(f)
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        # try trimming trailing comma
                        cleaned = line.rstrip(', ')
                        try:
                            results.append(json.loads(cleaned))
                        except Exception:
                            logger.debug(f"Skipping malformed line in {path}: {line[:200]}")
        return results
    except Exception as e:
        logger.error(f"Failed reading {path}: {e}")
        return []

# ----------------- normalize predictions -----------------
def _extract_score_from_risk_analysis(item: Dict[str, Any], score_name: str) -> float:
    """Try to extract a numeric score from risk_analysis.calculated_raw or risk_analysis.analyzed_scores."""
    ra = item.get('risk_analysis')
    if not isinstance(ra, dict):
        return np.nan
    # Try calculated_raw first (contains raw numeric scores)
    calc_raw = ra.get('calculated_raw', {})
    if isinstance(calc_raw, dict):
        val = calc_raw.get(score_name)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        # Also try nested dict with 'score' key (probabilistic format)
        entry = calc_raw.get(score_name)
        if isinstance(entry, dict):
            sc = entry.get('score')
            if sc is not None:
                try:
                    return float(sc)
                except (ValueError, TypeError):
                    pass
    # Try analyzed_scores
    analyzed = ra.get('analyzed_scores', {})
    if isinstance(analyzed, dict):
        val = analyzed.get(score_name) or analyzed.get(score_name.lower())
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return np.nan


def normalize_prediction(item: Dict[str,Any]) -> Dict[str,Any]:
    # flexible id keys
    hadm_id = item.get('hadm_id') or item.get('stay_id') or item.get('hadm') or item.get('id')
    try:
        hadm_id = int(hadm_id) if hadm_id is not None else 0
    except:
        hadm_id = 0

    vitals = item.get('extracted_vitals') or item.get('extracted_vital') or {}
    if not isinstance(vitals, dict):
        vitals = {}

    risk_text = item.get('risk_score') or item.get('risk') or item.get('risk_score_report') or ''

    rag_flag = item.get('rag_context_used')
    # some runs stored rag_context_used as string or missing
    if isinstance(rag_flag, str):
        rag_flag = rag_flag.lower() in ('true','1','yes')
    rag_flag = bool(rag_flag)

    # Extract scores from risk_score text via regex
    pred_news = extract_score_from_text(risk_text, NEWS_PATTERNS)
    pred_mews = extract_score_from_text(risk_text, MEWS_PATTERNS)

    # Fallback: try extracting from risk_analysis structured data (probabilistic format)
    if pd.isna(pred_news):
        pred_news = _extract_score_from_risk_analysis(item, 'NEWS')
    if pd.isna(pred_mews):
        pred_mews = _extract_score_from_risk_analysis(item, 'MEWS')

    return {
        'hadm_id': hadm_id,
        'Pred_SBP': normalize_value(vitals.get('sbp') or vitals.get('systolic') or vitals.get('systolic_bp')),
        'Pred_DBP': normalize_value(vitals.get('dbp') or vitals.get('diastolic')),
        'Pred_HR': normalize_value(vitals.get('heartrate') or vitals.get('hr')),
        'Pred_RR': normalize_value(vitals.get('resprate') or vitals.get('rr') or vitals.get('resp_rate')),
        'Pred_O2': normalize_value(vitals.get('o2sat') or vitals.get('spo2')),
        'Pred_Temp': normalize_value(vitals.get('temperature') or vitals.get('temperature_celsius')),
        'Pred_NEWS': pred_news,
        'Pred_MEWS': pred_mews,
        'rag_context_used': rag_flag
    }

# ----------------- load + join -----------------
def load_and_sanitize_data(gt_csv_path: str, experiments_dict: Dict[str,str]) -> Dict[str,pd.DataFrame]:
    logger.info(f"Loading GT from: {gt_csv_path}")
    try:
        # Auto-detect separator: try ';' first (common in pt-BR CSVs), fall back to ','
        df_gt = pd.read_csv(gt_csv_path, sep=';', na_values=['N/A','NA','None','null',''], low_memory=False)
        if len(df_gt.columns) <= 2:
            # ';' didn't split properly — retry with ','
            df_gt = pd.read_csv(gt_csv_path, sep=',', na_values=['N/A','NA','None','null',''], low_memory=False)
    except Exception as e:
        logger.error(f"Failed to read GT CSV: {e}")
        return {}

    # Drop spurious 'Unnamed' columns (from trailing separators)
    unnamed_cols = [c for c in df_gt.columns if c.startswith('Unnamed')]
    if unnamed_cols:
        df_gt = df_gt.drop(columns=unnamed_cols)

    id_col = 'hadm_id' if 'hadm_id' in df_gt.columns else ('stay_id' if 'stay_id' in df_gt.columns else None)
    if id_col is None:
        logger.error("Ground truth missing hadm_id/stay_id column.")
        return {}

    # Drop rows where id is NaN (empty rows from large CSVs)
    df_gt = df_gt.dropna(subset=[id_col])
    df_gt[id_col] = pd.to_numeric(df_gt[id_col], errors='coerce').fillna(0).astype(int)

    # Drop duplicate hadm_ids (keep first occurrence)
    before_dedup = len(df_gt)
    df_gt = df_gt.drop_duplicates(subset=[id_col], keep='first')
    if before_dedup != len(df_gt):
        logger.info(f"Removed {before_dedup - len(df_gt)} duplicate {id_col} rows from GT (kept first).")

    df_gt = df_gt.set_index(id_col)

    # apply normalization to GT candidate columns if present
    for key, candidates in GT_CANDIDATES.items():
        present = find_first_present(list(df_gt.columns), candidates)
        if present:
            df_gt[present] = df_gt[present].apply(normalize_value)

    merged_results = {}
    # also capture rag counts per experiment
    rag_summary = {}

    for label, path in experiments_dict.items():
        if not os.path.exists(path):
            logger.warning(f"Experiment file not found: {path}")
            continue
        raw = load_json_or_jsonl(path)
        preds = [normalize_prediction(it) for it in raw if it]
        if not preds:
            logger.warning(f"No usable predictions in {path}")
            continue
        df_pred = pd.DataFrame(preds).set_index('hadm_id')
        merged = df_gt.join(df_pred, how='inner')
        merged_results[label] = merged
        # rag summary
        rag_count = int(df_pred['rag_context_used'].sum()) if 'rag_context_used' in df_pred.columns else 0
        total_preds = len(df_pred)
        # Score extraction diagnostics
        news_extracted = int(df_pred['Pred_NEWS'].notna().sum()) if 'Pred_NEWS' in df_pred.columns else 0
        mews_extracted = int(df_pred['Pred_MEWS'].notna().sum()) if 'Pred_MEWS' in df_pred.columns else 0
        rag_summary[label] = {
            'raw_rows': len(raw),
            'joined_rows': len(merged),
            'pred_rows': total_preds,
            'rag_true': rag_count,
            'rag_pct': rag_count/total_preds if total_preds>0 else 0.0,
            'news_extracted': news_extracted,
            'mews_extracted': mews_extracted,
        }
        logger.info(f"Loaded {len(merged)} rows for experiment '{label}' (pred_rows={total_preds}, rag_true={rag_count}, NEWS_extracted={news_extracted}/{total_preds}, MEWS_extracted={mews_extracted}/{total_preds})")
    # attach rag_summary for debug/exports
    merged_results['_rag_summary'] = rag_summary
    return merged_results

# ----------------- hallucination detectors -----------------
def _compute_hallucination_for_vital(df: pd.DataFrame, pred_col: str, gt_col: str, tol: float) -> Tuple[float,int]:
    mask = df.get(gt_col) is not None and df.get(pred_col) is not None
    if not mask:
        # ensure we handle missing columns
        if gt_col not in df.columns or pred_col not in df.columns:
            return np.nan, 0
    mask = df[gt_col].notna() & df[pred_col].notna()
    if mask.sum() == 0:
        return np.nan, 0

    preds = df.loc[mask, pred_col].astype(float)
    gts = df.loc[mask, gt_col].astype(float)

    min_v, max_v = PHYSIO_RANGES.get(gt_col, (None, None))
    out_of_range = pd.Series(False, index=preds.index)
    if min_v is not None:
        out_of_range = out_of_range | (preds < min_v)
    if max_v is not None:
        out_of_range = out_of_range | (preds > max_v)

    abs_diff = (gts - preds).abs()
    extreme_abs = abs_diff > (5.0 * tol)
    rel_err = abs_diff / (gts.abs() + 1e-6)
    extreme_rel = rel_err > RELATIVE_ERROR_THRESHOLD

    pred_std = preds.std(ddof=0)
    const_pred_flag = False
    if pred_std <= CONST_VALUE_SUSPECT_STD and mask.sum() >= 8:
        mean_diff = abs((gts - preds).mean())
        const_pred_flag = mean_diff > (3.0 * tol)

    if const_pred_flag:
        const_mask = abs_diff > (3.0 * tol)
    else:
        const_mask = pd.Series(False, index=preds.index)

    halluc_mask = out_of_range | extreme_abs | extreme_rel | const_mask
    halluc_rate = float(halluc_mask.sum()) / len(preds)
    return halluc_rate, int(len(preds))

def _compute_hallucination_for_score(df: pd.DataFrame, pred_col: str, gt_col: str) -> Tuple[float,int]:
    if gt_col not in df.columns or pred_col not in df.columns:
        return np.nan, 0
    mask = df[gt_col].notna() & df[pred_col].notna()
    if mask.sum() == 0:
        return np.nan, 0
    preds = df.loc[mask, pred_col].astype(float)
    gts = df.loc[mask, gt_col].astype(float)
    out_of_range = (preds < 0) | (preds > 20)
    abs_diff = (preds - gts).abs()
    extreme = abs_diff >= SCORE_EXTREME_DIFF_THRESHOLD
    rel_err = abs_diff / (gts.replace(0,1).abs())
    extreme_rel = rel_err > RELATIVE_ERROR_THRESHOLD
    halluc_mask = out_of_range | extreme | extreme_rel
    halluc_rate = float(halluc_mask.sum()) / len(preds)
    return halluc_rate, int(len(preds))


# ----------------- bootstrap & significance helpers -----------------
def _bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
    arr = np.array([v for v in values if not pd.isna(v)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    point = float(arr.mean())
    if arr.size == 1:
        return point, point, point
    boots = []
    n = arr.size
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        boots.append(sample.mean())
    boots = np.sort(boots)
    lower = float(np.percentile(boots, 100*alpha/2))
    upper = float(np.percentile(boots, 100*(1-alpha/2)))
    return point, lower, upper


def _halluc_flag_vital(gt: float, pred: float, tol: float, gt_key: str) -> bool:
    min_v, max_v = PHYSIO_RANGES.get(gt_key, (None, None))
    abs_diff = abs(gt - pred)
    rel_err = abs_diff / (abs(gt) + 1e-6)
    out_of_range = (min_v is not None and pred < min_v) or (max_v is not None and pred > max_v)
    extreme_abs = abs_diff > (5.0 * tol)
    extreme_rel = rel_err > RELATIVE_ERROR_THRESHOLD
    return bool(out_of_range or extreme_abs or extreme_rel)


def _halluc_flag_score(gt: float, pred: float) -> bool:
    out_of_range = (pred < 0) or (pred > 20)
    abs_diff = abs(pred - gt)
    extreme = abs_diff >= SCORE_EXTREME_DIFF_THRESHOLD
    rel_err = abs_diff / (abs(gt) + 1e-6)
    extreme_rel = rel_err > RELATIVE_ERROR_THRESHOLD
    return bool(out_of_range or extreme or extreme_rel)


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact binomial p-value for McNemar (two-sided)."""
    n = b + c
    if n == 0:
        return np.nan
    k = min(b, c)
    p = 0.0
    for i in range(0, k+1):
        p += comb(n, i) * (0.5 ** n)
    p = 2 * p
    return float(min(1.0, p))

# ----------------- metrics calculation -----------------
def calculate_clinical_metrics_improved(merged_results: Dict[str,pd.DataFrame]) -> pd.DataFrame:
    rows = []
    vitals_targets = [
        ('sbp', 'Pred_SBP', 'SBP', 10.0),
        ('dbp', 'Pred_DBP', 'DBP', 10.0),
        ('heartrate', 'Pred_HR', 'HR', 5.0),
        ('resprate', 'Pred_RR', 'RR', 2.0),
        ('o2sat', 'Pred_O2', 'O2', 2.0),
        ('temperature_celsius', 'Pred_Temp', 'Temp', 0.5),
    ]
    score_targets = [
        ('mews', 'Pred_MEWS', 'MEWS'),
        ('news', 'Pred_NEWS', 'NEWS'),
    ]

    # iterate systems (skip rag_summary key)
    for system, df in merged_results.items():
        if system == '_rag_summary':
            continue
        if df is None or df.empty:
            continue
        # for each GT candidate mapping, find actual column name in df
        for gt_key, pred_col, label, tol in vitals_targets:
            # find a GT column present among candidates
            gt_col = find_first_present(list(df.columns), GT_CANDIDATES.get(gt_key, [gt_key]))
            if not gt_col or pred_col not in df.columns:
                continue
            mask = df[gt_col].notna() & df[pred_col].notna()
            if mask.sum() == 0:
                continue
            diff = (df.loc[mask, gt_col] - df.loc[mask, pred_col]).abs()
            within_tol = diff <= tol
            # Trata "acerto dentro da tolerância" como classe positiva; não há classes preditas separadas.
            # Portanto, precision = recall = taxa de acerto dentro da tolerância; F1 = mesmo valor.
            precision = recall = float(within_tol.mean()) if len(within_tol) else np.nan
            clin_f1 = precision
            halluc_rate, n_pred = _compute_hallucination_for_vital(df, pred_col, gt_col, tol)
            rows.append({
                'System': system,
                'Variable': label,
                'Metric_Type': 'Clin_F1',
                'Clinical_Score': clin_f1,
                'Precision': precision,
                'Recall': recall,
                'Strict_Match': float((diff == 0).mean()) if len(diff) else np.nan,
                'MAE': float(diff.mean()) if len(diff) else np.nan,
                'Halluc_Rate': halluc_rate,
                'Samples': int(mask.sum())
            })

        # scores
        for gt_key, pred_col, label in score_targets:
            gt_col = find_first_present(list(df.columns), GT_CANDIDATES.get(gt_key, [gt_key]))
            if not gt_col or pred_col not in df.columns:
                continue
            mask = df[gt_col].notna() & df[pred_col].notna()
            if mask.sum() == 0:
                continue
            if label == 'MEWS':
                gt_risk = df.loc[mask, gt_col].apply(get_mews_risk)
                pred_risk = df.loc[mask, pred_col].apply(get_mews_risk)
            else:
                gt_risk = df.loc[mask, gt_col].apply(get_news_risk)
                pred_risk = df.loc[mask, pred_col].apply(get_news_risk)
            valid = gt_risk.notna() & pred_risk.notna()
            if valid.sum() == 0:
                continue
            acc = float((gt_risk[valid] == pred_risk[valid]).mean())
            labels = ['Low','Medium','High']
            try:
                precision = float(precision_score(gt_risk[valid], pred_risk[valid], labels=labels, average='macro', zero_division=0))
                recall = float(recall_score(gt_risk[valid], pred_risk[valid], labels=labels, average='macro', zero_division=0))
                f1 = float(f1_score(gt_risk[valid], pred_risk[valid], labels=labels, average='macro', zero_division=0))
            except Exception:
                precision = recall = f1 = np.nan
            halluc_rate, n_pred_score = _compute_hallucination_for_score(df, pred_col, gt_col)
            rows.append({
                'System': system,
                'Variable': label,
                'Metric_Type': 'Risk_Acc',
                'Clinical_Score': acc,
                'Precision': precision,
                'Recall': recall,
                'Strict_Match': float((df.loc[mask, gt_col] == df.loc[mask, pred_col]).mean()),
                'MAE': float((df.loc[mask, gt_col] - df.loc[mask, pred_col]).abs().mean()),
                'Halluc_Rate': halluc_rate,
                'Samples': int(valid.sum())
            })

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        for c in ['Clinical_Score','Precision','Recall','Strict_Match','MAE','Halluc_Rate']:
            if c in metrics_df.columns:
                metrics_df[c] = pd.to_numeric(metrics_df[c], errors='coerce')
        metrics_df = metrics_df.sort_values(['System','Variable','Metric_Type']).reset_index(drop=True)
    return metrics_df


def _build_sample_records(merged_results: Dict[str,pd.DataFrame]) -> List[Dict[str,Any]]:
    records = []
    vitals_targets = [
        ('sbp', 'Pred_SBP', 'SBP', 10.0),
        ('dbp', 'Pred_DBP', 'DBP', 10.0),
        ('heartrate', 'Pred_HR', 'HR', 5.0),
        ('resprate', 'Pred_RR', 'RR', 2.0),
        ('o2sat', 'Pred_O2', 'O2', 2.0),
        ('temperature_celsius', 'Pred_Temp', 'Temp', 0.5),
    ]

    for system, df in merged_results.items():
        if system == '_rag_summary' or df is None or df.empty:
            continue
        for gt_key, pred_col, label, tol in vitals_targets:
            gt_col = find_first_present(list(df.columns), GT_CANDIDATES.get(gt_key, [gt_key]))
            if not gt_col or pred_col not in df.columns:
                continue
            for hadm_id, row in df[[gt_col, pred_col]].iterrows():
                gt = row[gt_col]
                pred = row[pred_col]
                parsed = pd.notna(gt) and pd.notna(pred)
                within = None
                halluc = None
                if parsed:
                    gt_f = float(gt)
                    pred_f = float(pred)
                    within = abs(gt_f - pred_f) <= tol
                    halluc = _halluc_flag_vital(gt_f, pred_f, tol, gt_key)
                records.append({
                    'system': system,
                    'hadm_id': hadm_id,
                    'variable': label,
                    'parsed': parsed,
                    'within_tol': within,
                    'halluc': halluc
                })

        # NEWS risk correctness for McNemar
        news_gt_col = find_first_present(list(df.columns), GT_CANDIDATES.get('news', ['news']))
        if news_gt_col and 'Pred_NEWS' in df.columns:
            for hadm_id, row in df[[news_gt_col, 'Pred_NEWS']].iterrows():
                gt = row[news_gt_col]
                pred = row['Pred_NEWS']
                parsed = pd.notna(gt) and pd.notna(pred)
                news_correct = None
                if parsed:
                    gt_label = get_news_risk(gt)
                    pred_label = get_news_risk(pred)
                    news_correct = (gt_label is not None and pred_label is not None and gt_label == pred_label)
                records.append({
                    'system': system,
                    'hadm_id': hadm_id,
                    'variable': 'NEWS',
                    'parsed': parsed,
                    'within_tol': news_correct,
                    'halluc': None,
                    'is_news': True
                })
    return records

# ----------------- fallback compact builder -----------------
def build_compact_metrics_from_merged(merged_results: Dict[str,pd.DataFrame]) -> pd.DataFrame:
    rows = []
    vitals_targets = [
        ('sbp', 'Pred_SBP', 'SBP', 10.0),
        ('dbp', 'Pred_DBP', 'DBP', 10.0),
        ('heartrate', 'Pred_HR', 'HR', 5.0),
        ('resprate', 'Pred_RR', 'RR', 2.0),
        ('o2sat', 'Pred_O2', 'O2', 2.0),
        ('temperature_celsius', 'Pred_Temp', 'Temp', 0.5),
    ]
    score_targets = [
        ('mews', 'Pred_MEWS', 'MEWS'),
        ('news', 'Pred_NEWS', 'NEWS'),
    ]
    for system, df in merged_results.items():
        if system == '_rag_summary':
            continue
        if df is None or df.empty:
            continue
        for gt_key, pred_col, label, tol in vitals_targets:
            gt_col = find_first_present(list(df.columns), GT_CANDIDATES.get(gt_key, [gt_key]))
            if not gt_col or pred_col not in df.columns:
                continue
            mask = df[gt_col].notna() & df[pred_col].notna()
            if mask.sum() == 0:
                continue
            mae = float((df.loc[mask, gt_col] - df.loc[mask, pred_col]).abs().mean())
            halluc_rate, n = _compute_hallucination_for_vital(df, pred_col, gt_col, tol)
            rows.append({'System': system, 'Variable': label, 'MAE': mae, 'Halluc_Rate': halluc_rate, 'Samples': int(mask.sum())})
        for gt_key, pred_col, label in score_targets:
            gt_col = find_first_present(list(df.columns), GT_CANDIDATES.get(gt_key, [gt_key]))
            if not gt_col or pred_col not in df.columns:
                continue
            mask = df[gt_col].notna() & df[pred_col].notna()
            if mask.sum() == 0:
                continue
            mae = float((df.loc[mask, gt_col] - df.loc[mask, pred_col]).abs().mean())
            halluc_rate, n = _compute_hallucination_for_score(df, pred_col, gt_col)
            rows.append({'System': system, 'Variable': label, 'MAE': mae, 'Halluc_Rate': halluc_rate, 'Samples': int(mask.sum())})
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=['System','Variable','MAE','Halluc_Rate','Samples'])


def _generate_significance_report(records: List[Dict[str,Any]], out_dir: str = "results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, "tcc_significance.md")

    df = pd.DataFrame(records)
    if df.empty:
        with open(fname, 'w', encoding='utf-8') as f:
            f.write("Sem dados para intervalos de confiança.")
        return fname

    # Drop NEWS helper rows for general metrics
    vit_df = df[df['variable'] != 'NEWS']

    rows_ci = []
    for system, g in vit_df.groupby('system'):
        within_vals = [v for v in g['within_tol'] if v is not None]
        halluc_vals = [v for v in g['halluc'] if v is not None]
        parsed_vals = g['parsed'].astype(float).tolist()

        f1_point, f1_lo, f1_hi = _bootstrap_ci(within_vals)
        h_point, h_lo, h_hi = _bootstrap_ci(halluc_vals)
        p_point, p_lo, p_hi = _bootstrap_ci(parsed_vals)

        rows_ci.append({
            'System': system,
            'Macro_F1_mean': f1_point,
            'Macro_F1_CI95': f"[{f1_lo:.3f}, {f1_hi:.3f}]" if not np.isnan(f1_lo) else "NA",
            'Halluc_Rate_mean': h_point,
            'Halluc_Rate_CI95': f"[{h_lo:.3f}, {h_hi:.3f}]" if not np.isnan(h_lo) else "NA",
            'Parsing_Success_mean': p_point,
            'Parsing_Success_CI95': f"[{p_lo:.3f}, {p_hi:.3f}]" if not np.isnan(p_lo) else "NA",
        })

    ci_table = pd.DataFrame(rows_ci)

    # Paired tests TA vs B0
    paired_rows = []
    def _paired_mcnemar(var_label: str, filter_news: bool=False):
        # Build per-item correctness for TA and B0
        if filter_news:
            sub = df[df.get('is_news') == True]
        else:
            sub = df[df['variable'] == var_label]
        ta = sub[sub['system']=='TriaLogic (Agents)']
        b0 = sub[sub['system']=='Baseline (Zero-Shot)']
        if ta.empty or b0.empty:
            return None
        merged = ta.merge(b0, on=['hadm_id','variable'], suffixes=('_ta','_b0'))
        merged = merged[(merged['parsed_ta']) & (merged['parsed_b0']) & merged['within_tol_ta'].notna() & merged['within_tol_b0'].notna()]
        if merged.empty:
            return None
        a_correct = merged['within_tol_ta'].astype(bool)
        b_correct = merged['within_tol_b0'].astype(bool)
        b_count = int((a_correct & ~b_correct).sum())
        c_count = int((~a_correct & b_correct).sum())
        pval = _mcnemar_exact(b_count, c_count)
        return {
            'Test': f"McNemar TA vs B0 ({var_label})",
            'n_pairs': len(merged),
            'b_TA_correct_B0_wrong': b_count,
            'c_TA_wrong_B0_correct': c_count,
            'p_value': pval
        }

    # Entity-level: aggregate all vitals
    entity_sub = df[(df['variable'] != 'NEWS')]
    ta_e = entity_sub[entity_sub['system']=='TriaLogic (Agents)']
    b0_e = entity_sub[entity_sub['system']=='Baseline (Zero-Shot)']
    if not ta_e.empty and not b0_e.empty:
        merged_e = ta_e.merge(b0_e, on=['hadm_id','variable'], suffixes=('_ta','_b0'))
        merged_e = merged_e[(merged_e['parsed_ta']) & (merged_e['parsed_b0']) & merged_e['within_tol_ta'].notna() & merged_e['within_tol_b0'].notna()]
        if not merged_e.empty:
            a_correct = merged_e['within_tol_ta'].astype(bool)
            b_correct = merged_e['within_tol_b0'].astype(bool)
            b_count = int((a_correct & ~b_correct).sum())
            c_count = int((~a_correct & b_correct).sum())
            pval = _mcnemar_exact(b_count, c_count)
            paired_rows.append({
                'Test': 'McNemar TA vs B0 (Entity-level extraction)',
                'n_pairs': len(merged_e),
                'b_TA_correct_B0_wrong': b_count,
                'c_TA_wrong_B0_correct': c_count,
                'p_value': pval
            })

    news_row = _paired_mcnemar('NEWS', filter_news=True)
    if news_row:
        paired_rows.append(news_row)

    with open(fname, 'w', encoding='utf-8') as f:
        f.write("# Significance & Confidence Intervals\n\n")
        if not ci_table.empty:
            f.write("## Bootstrap 95% CI (2k amostras)\n\n")
            f.write(ci_table.round(3).to_markdown(index=False))
            f.write("\n\n")
        if paired_rows:
            f.write("## Paired tests (McNemar, exato) TA vs B0\n\n")
            f.write(pd.DataFrame(paired_rows).round(4).to_markdown(index=False))
            f.write("\n")
        else:
            f.write("Sem pares suficientes para McNemar.\n")

    # print to terminal
    print("\n=== Significance (bootstrap CI + McNemar) ===")
    if not ci_table.empty:
        print(ci_table.round(3).to_string(index=False))
    if paired_rows:
        print("\n-- McNemar TA vs B0 --")
        print(pd.DataFrame(paired_rows).round(4).to_string(index=False))

    return fname


# ----------------- exports and pretty printing -----------------
def export_pretty_tables(metrics_df: pd.DataFrame, merged_results: Dict[str,pd.DataFrame], out_dir: str = "results") -> Dict[str,str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "tcc_final_metrics.csv")
    try:
        metrics_df.to_csv(csv_path, index=False)
    except Exception:
        pd.DataFrame().to_csv(csv_path)

    fallback_compact = None
    if metrics_df.empty:
        logger.warning("metrics_df vazio — usando fallback compacto direto de merged_results.")
        fallback_compact = build_compact_metrics_from_merged(merged_results)
        if not fallback_compact.empty:
            fallback_compact.to_csv(os.path.join(out_dir, "tcc_compact_fallback.csv"), index=False)

    if not metrics_df.empty and 'System' in metrics_df.columns:
        source_df = metrics_df
    else:
        source_df = fallback_compact if fallback_compact is not None else pd.DataFrame()

    # summary aggregated by system
    summary_rows = []
    if not source_df.empty:
        for system, g in source_df.groupby('System'):
            total_samples = int(g['Samples'].sum()) if 'Samples' in g.columns else 0
            weighted_halluc = float((g['Halluc_Rate'] * g['Samples']).sum()/total_samples) if total_samples>0 else np.nan
            summary_rows.append({
                'System': system,
                'Avg_MAE': float(g['MAE'].mean()) if 'MAE' in g.columns else np.nan,
                'Avg_Halluc_Rate': weighted_halluc,
                'Total_Samples': total_samples
            })
    summary_df = pd.DataFrame(summary_rows).sort_values('System') if summary_rows else pd.DataFrame()

    tex_path = os.path.join(out_dir, "tcc_final_metrics.tex")
    try:
        to_write = metrics_df if not metrics_df.empty else fallback_compact
        if to_write is None or to_write.empty:
            pd.DataFrame(columns=['System','Variable']).to_latex(tex_path, index=False)
        else:
            to_write.round(4).to_latex(tex_path, index=False, longtable=True, caption="TCC: Vital signs vs Clinical Scores", float_format="%.4f", na_rep="NA")
    except Exception as e:
        logger.warning(f"Erro LaTeX: {e}")

    html_path = os.path.join(out_dir, "tcc_final_metrics.html")
    try:
        if not source_df.empty:
            pivot_mae = source_df.pivot_table(index='Variable', columns='System', values='MAE', aggfunc='mean')
            pivot_h = source_df.pivot_table(index='Variable', columns='System', values='Halluc_Rate', aggfunc='mean')
            combined = pd.concat([pivot_mae.add_suffix(" (MAE)"), pivot_h.add_suffix(" (Halluc)")], axis=1)
            combined = combined.reindex(sorted(combined.columns), axis=1)  # nicer ordering
            # use Styler.to_html() for compatibility
            html = combined.round(3).fillna("NA").to_html()
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
        else:
            pd.DataFrame().to_html(html_path)
    except Exception as e:
        logger.warning(f"Erro HTML consolidado: {e}")

    md_path = os.path.join(out_dir, "tcc_final_metrics.md")
    try:
        if not source_df.empty:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(source_df.round(4).to_markdown(index=False))
    except Exception as e:
        logger.warning(f"Erro Markdown: {e}")

    summary_csv = os.path.join(out_dir, "tcc_metrics_summary.csv")
    try:
        summary_df.to_csv(summary_csv, index=False)
    except Exception:
        pd.DataFrame().to_csv(summary_csv)

    per_sys_dir = os.path.join(out_dir, "per_system")
    os.makedirs(per_sys_dir, exist_ok=True)
    if not source_df.empty:
        for system, g in source_df.groupby('System'):
            safe_name = str(system)
            if hasattr(safe_name, 'replace'):
                safe_name = safe_name.replace(' ', '_')
            fname = os.path.join(per_sys_dir, f"{safe_name}.html")
            try:
                disp = g[['Variable','MAE','Halluc_Rate','Samples']].round(4)
                disp.to_html(fname, index=False)
            except Exception as e:
                logger.warning(f"Could not write per-system HTML for {system}: {e}")

    # Terminal output: compact summary + consolidated pivot print
    print("\n=== Resumo agregado por Sistema (terminal) ===")
    if not summary_df.empty:
        print(summary_df.round(4).to_string(index=False))
    else:
        print("Nenhum resumo agregado disponível (summary_df vazio).")
        print("\n--- Diagnóstico rápido (merged_results) ---")
        for system, df in merged_results.items():
            if system == '_rag_summary': continue
            print(f"System: {system} | rows: {len(df)} | cols: {len(df.columns)}")
            print("  Cols sample:", list(df.columns)[:15])

    if not source_df.empty:
        pivot_mae = source_df.pivot_table(index='Variable', columns='System', values='MAE', aggfunc='mean')
        pivot_h = source_df.pivot_table(index='Variable', columns='System', values='Halluc_Rate', aggfunc='mean')
        combined = pd.concat([pivot_mae.add_suffix(" (MAE)"), pivot_h.add_suffix(" (Halluc)")], axis=1)
        combined = combined.reindex(sorted(combined.columns), axis=1)

        # Versão compacta para terminal (colunas mais curtas, largura maior, 3 casas decimais)
        short_map = {
            'Baseline (Zero-Shot)': 'B0',
            'Baseline (One-Shot)': 'B1',
            'No RAG': 'NR',
            'TriaLogic (Agents)': 'TA',
            'TriaLogic (No Validator)': 'TNV',
            'TriaLogic (Probabilistic)': 'TP'
        }
        def _shorten(col):
            for long, short in short_map.items():
                if col.startswith(long):
                    return col.replace(long, short)
            return col

        combined_short = combined.copy()
        combined_short.columns = [_shorten(c) for c in combined_short.columns]
        with pd.option_context('display.width', 200, 'display.max_columns', None):
            print("\n--- Tabela consolidada (MAE + Hallucination Rate) ---")
            print(combined_short.round(3).fillna("NA").to_string())
    else:
        print("\n(sem tabela consolidada - sem dados)")

    return {
        "csv": csv_path,
        "latex": tex_path,
        "html": html_path,
        "markdown": md_path,
        "summary_csv": summary_csv,
        "per_system_dir": per_sys_dir
    }


# ----------------- Article-ready summary -----------------
def build_article_report(metrics_df: pd.DataFrame, exec_seconds: Optional[float] = None, app_seconds: Optional[float] = None, out_dir: str = "results") -> str:
    """
    Gera tabelas compactas (Markdown) com F1/Precision/Recall/MAE/Halluc para vitais
    e Acc/Precision/Recall/MAE/Halluc para scores de risco, além do tempo de execução.
    Retorna o caminho do arquivo gerado.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, "tcc_article_tables.md")

    if metrics_df.empty:
        with open(fname, 'w', encoding='utf-8') as f:
            msg = "Nenhum dado disponível."
            if exec_seconds is not None:
                msg += " Tempo (avaliação): {:.2f}s.".format(exec_seconds)
            if app_seconds is not None:
                msg += " Tempo (TriaLogic): {:.2f}s.".format(app_seconds)
            f.write(msg)
        return fname

    vitals = metrics_df[metrics_df['Metric_Type'] == 'Clin_F1'][['System','Variable','Clinical_Score','Precision','Recall','MAE','Halluc_Rate','Samples']]
    scores = metrics_df[metrics_df['Metric_Type'] == 'Risk_Acc'][['System','Variable','Clinical_Score','Precision','Recall','MAE','Halluc_Rate','Samples']]

    def _fmt(df):
        return df.copy().assign(
            Clinical_Score=lambda d: d['Clinical_Score'].round(3),
            Precision=lambda d: d['Precision'].round(3),
            Recall=lambda d: d['Recall'].round(3),
            MAE=lambda d: d['MAE'].round(3),
            Halluc_Rate=lambda d: d['Halluc_Rate'].round(3)
        )

    vitals_fmt = _fmt(vitals)
    scores_fmt = _fmt(scores)

    lines = []
    lines.append("# Tabelas para artigo\n")
    if exec_seconds is not None:
        lines.append(f"Tempo (avaliação): {exec_seconds:.2f}s")
    if app_seconds is not None:
        lines.append(f"Tempo (TriaLogic): {app_seconds:.2f}s")
    lines.append("")

    if not vitals_fmt.empty:
        lines.append("## Vitals – F1/Prec/Rec/MAE/Halluc\n")
        lines.append(vitals_fmt.to_markdown(index=False))
        lines.append("\n")
    if not scores_fmt.empty:
        lines.append("## Scores de risco – Acc/Prec/Rec/MAE/Halluc\n")
        lines.append(scores_fmt.to_markdown(index=False))
        lines.append("\n")

    with open(fname, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    # Também imprimir versão enxuta no terminal
    print("\n=== Formato artigo (resumo) ===")
    if not vitals_fmt.empty:
        print("-- Vitals --")
        print(vitals_fmt.to_string(index=False))
    if not scores_fmt.empty:
        print("\n-- Scores de risco --")
        print(scores_fmt.to_string(index=False))
    if exec_seconds is not None:
        print(f"\nTempo (avaliação): {exec_seconds:.2f}s")
    if app_seconds is not None:
        print(f"Tempo (TriaLogic): {app_seconds:.2f}s")

    return fname

# ----------------- ENTRYPOINT -----------------
if __name__ == "__main__":
    t0 = time.perf_counter()

    BASE_DIR = os.getcwd()
    GT_PATH = os.path.join(BASE_DIR, "data", "ground_truth.csv")
    EXPERIMENTS = {
        "Baseline (Zero-Shot)": os.path.join(BASE_DIR, "results", "baseline_results.jsonl"),
        "Baseline (One-Shot)": os.path.join(BASE_DIR, "results", "oneshot_baseline_results.jsonl"),
        "No RAG": os.path.join(BASE_DIR, "results", "norag_experiment_results_v1.jsonl"),
        "TriaLogic (Agents)": os.path.join(BASE_DIR, "results", "experiment_results_v1.jsonl"),
        "TriaLogic (No Validator)": os.path.join(BASE_DIR, "results", "novalidation_experiment_results_v1.jsonl"),
        "TriaLogic (Probabilistic)": os.path.join(BASE_DIR, "results", "probabilistic_experiment_results_v1.jsonl"),
    }

    if not os.path.exists(GT_PATH):
        logger.error(f"Ground truth not found at: {GT_PATH}")
        raise SystemExit(1)

    merged = load_and_sanitize_data(GT_PATH, EXPERIMENTS)
    if not merged:
        logger.error("No merged results. Aborting.")
        raise SystemExit(1)

    metrics_df = calculate_clinical_metrics_improved(merged)
    sample_records = _build_sample_records(merged)
    exports = export_pretty_tables(metrics_df, merged_results=merged, out_dir=os.path.join(BASE_DIR, "results"))

    t1 = time.perf_counter()
    exec_seconds = t1 - t0

    # tempo específico do TriaLogic (se fornecido pelo usuário, p.ex. variável de ambiente)
    app_seconds = None
    env_runtime = os.getenv("TRIALOGIC_RUNTIME_SECONDS")
    if env_runtime:
        try:
            app_seconds = float(env_runtime)
        except ValueError:
            logger.warning("TRIALOGIC_RUNTIME_SECONDS não pôde ser convertido para float.")

    article_path = build_article_report(metrics_df, exec_seconds=exec_seconds, app_seconds=app_seconds, out_dir=os.path.join(BASE_DIR, "results"))
    signif_path = _generate_significance_report(sample_records, out_dir=os.path.join(BASE_DIR, "results"))

    logger.info("Exports written:")
    for k,v in exports.items():
        logger.info(f" - {k}: {v}")
    logger.info(f" - article: {article_path}")
    logger.info(f" - significance: {signif_path}")
    logger.info(f" - execution_seconds: {exec_seconds:.2f}")

    # print RAG summary if available
    rag_summary = merged.get('_rag_summary', {})
    if isinstance(rag_summary, dict) and len(rag_summary) > 0:
        print("\n=== RAG & Score Extraction SUMMARY per Experiment ===")
        for k, s in rag_summary.items():
            news_ext = s.get('news_extracted', '?')
            mews_ext = s.get('mews_extracted', '?')
            print(f"{k}: joined={s['joined_rows']}, preds={s['pred_rows']}, rag_true={s['rag_true']} ({s['rag_pct']:.2%}), NEWS_extracted={news_ext}/{s['pred_rows']}, MEWS_extracted={mews_ext}/{s['pred_rows']}")

    print("\n✅ Avaliação concluída. Verifique 'results/' e 'results/per_system/' para as tabelas coloridas.")
