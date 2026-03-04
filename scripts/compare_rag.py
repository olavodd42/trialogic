"""Qualitative and quantitative comparison of RAG vs No-RAG experiment outputs.

Produces:
  1. Quantitative RAG-retrieval metrics (overall + per-cohort)
  2. Qualitative case-by-case classification (safety wins, context wins, etc.)
"""

import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

# Path configuration
PATH_NO_RAG = "results/norag_experiment_results_v1.jsonl" 
PATH_WITH_RAG = "results/experiment_results_v1.jsonl"
PATH_METRICS_CSV = "results/rag_quantitative_metrics.csv"

# ── Thresholds for classification ──
_MIN_QUOTE_LEN = 10   # chars – evidence_quote is "substantive"
_MIN_REF_LEN   = 3    # chars – protocol_reference is present


def load_jsonl(path: str) -> Dict[str, Any]:
    """Load a JSONL file and index successful records by subject+hadm key."""
    data = {}
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return data
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                key = f"{record.get('subject_id')}_{record.get('hadm_id')}"
                if "error" not in record:
                    data[key] = record
            except:
                pass
    return data


def load_jsonl_all(path: str) -> List[Dict[str, Any]]:
    """Load every record from a JSONL (including errors), as a list."""
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# ───────────────────────────────────────────────────────
#  NEW – Quantitative RAG-retrieval metrics
# ───────────────────────────────────────────────────────
def _pct(num: int, den: int) -> str:
    """Format a percentage safely."""
    return f"{100 * num / den:.1f}%" if den else "N/A"


def quantitative_rag_metrics() -> pd.DataFrame:
    """Compute and print quantitative RAG-retrieval metrics.

    Metrics produced (overall + per-cohort):
      • rag_retrieval_rate – cases where rag_context_used == True
      • protocol_hit_rate – cases where auditor returned a protocol_reference
      • evidence_yield     – cases with a substantive evidence_quote (>10 chars)
      • suggestion_rate    – cases with a non-trivial clinical_suggestion (>10 chars)

    Also computes the *same* evidence_yield for the No-RAG baseline so the
    uplift is directly comparable.

    Returns a DataFrame and saves it as CSV for inclusion in the article.
    """
    rag_records = load_jsonl_all(PATH_WITH_RAG)
    norag_records = load_jsonl_all(PATH_NO_RAG)

    # ── Report processing errors first ──
    rag_errors = [r for r in rag_records if "error" in r]
    norag_errors = [r for r in norag_records if "error" in r]

    # ── helpers ──
    def _classify(records: List[dict]) -> Dict[str, Dict[str, int]]:
        """Return per-cohort counts dict[cohort][metric] = count."""
        stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in records:
            if "error" in r:
                continue
            cohort = r.get("cohort", "unknown")
            stats[cohort]["total"] += 1

            if r.get("rag_context_used") is True:
                stats[cohort]["rag_used"] += 1

            av = r.get("auditor_verdict") or {}
            if not isinstance(av, dict):
                continue

            pr = str(av.get("protocol_reference", "")).strip()
            eq = str(av.get("evidence_quote", "")).strip()
            cs = str(av.get("clinical_suggestion", "")).strip()

            if len(pr) >= _MIN_REF_LEN:
                stats[cohort]["has_ref"] += 1
            if len(eq) > _MIN_QUOTE_LEN:
                stats[cohort]["has_quote"] += 1
            if len(cs) > _MIN_QUOTE_LEN:
                stats[cohort]["has_suggestion"] += 1
        return stats

    rag_stats = _classify(rag_records)
    norag_stats = _classify(norag_records)

    n_rag_success = sum(s["total"] for s in rag_stats.values())
    n_norag_success = sum(s["total"] for s in norag_stats.values())

    # ── Pretty-print ──
    print("\n" + "=" * 72)
    print("QUANTITATIVE RAG-RETRIEVAL METRICS")
    print("=" * 72)
    print(f"\nTotal records:  RAG = {len(rag_records)}  |  No-RAG = {len(norag_records)}")
    if rag_errors or norag_errors:
        print(f"  Processing errors excluded:  RAG = {len(rag_errors)}  |  No-RAG = {len(norag_errors)}")
        if rag_errors:
            error_hadms = [str(r.get("hadm_id", "?")) for r in rag_errors]
            print(f"  RAG error hadm_ids: {', '.join(error_hadms)}")
    print(f"  Effective N:  RAG = {n_rag_success}  |  No-RAG = {n_norag_success}")

    rows = []  # for CSV / DataFrame

    for label, stats, is_rag in [("WITH RAG", rag_stats, True),
                                  ("WITHOUT RAG (baseline)", norag_stats, False)]:
        print(f"\n── {label} ──")
        header = (
            f"{'Cohort':<16} {'N':>4}  "
            + (f"{'RAG used':>10}  " if is_rag else "")
            + f"{'Proto Ref':>10}  {'Evid Quote':>11}  {'Suggestion':>11}"
        )
        print(header)
        print("-" * len(header))

        all_total = all_rag = all_ref = all_quote = all_sug = 0
        for cohort in sorted(stats.keys()):
            s = stats[cohort]
            t = s["total"]
            all_total += t
            all_rag += s["rag_used"]
            all_ref += s["has_ref"]
            all_quote += s["has_quote"]
            all_sug += s["has_suggestion"]

            line = f"{cohort:<16} {t:>4}  "
            if is_rag:
                line += f"{s['rag_used']:>4} ({_pct(s['rag_used'], t):>6})  "
            line += (
                f"{s['has_ref']:>4} ({_pct(s['has_ref'], t):>6})  "
                f"{s['has_quote']:>4} ({_pct(s['has_quote'], t):>6})  "
                f"{s['has_suggestion']:>4} ({_pct(s['has_suggestion'], t):>6})"
            )
            print(line)

            rows.append({
                "system": "TriaLogic (RAG)" if is_rag else "No-RAG Baseline",
                "cohort": cohort,
                "n": t,
                "rag_retrieval_rate": round(s["rag_used"] / t, 4) if t and is_rag else None,
                "protocol_hit_rate": round(s["has_ref"] / t, 4) if t else None,
                "evidence_yield": round(s["has_quote"] / t, 4) if t else None,
                "suggestion_rate": round(s["has_suggestion"] / t, 4) if t else None,
            })

        # total row
        print("-" * len(header))
        total_line = f"{'TOTAL':<16} {all_total:>4}  "
        if is_rag:
            total_line += f"{all_rag:>4} ({_pct(all_rag, all_total):>6})  "
        total_line += (
            f"{all_ref:>4} ({_pct(all_ref, all_total):>6})  "
            f"{all_quote:>4} ({_pct(all_quote, all_total):>6})  "
            f"{all_sug:>4} ({_pct(all_sug, all_total):>6})"
        )
        print(total_line)

        rows.append({
            "system": "TriaLogic (RAG)" if is_rag else "No-RAG Baseline",
            "cohort": "TOTAL",
            "n": all_total,
            "rag_retrieval_rate": round(all_rag / all_total, 4) if all_total and is_rag else None,
            "protocol_hit_rate": round(all_ref / all_total, 4) if all_total else None,
            "evidence_yield": round(all_quote / all_total, 4) if all_total else None,
            "suggestion_rate": round(all_sug / all_total, 4) if all_total else None,
        })

    # ── Uplift summary ──
    rag_t = sum(s["total"] for s in rag_stats.values())
    rag_q = sum(s["has_quote"] for s in rag_stats.values())
    norag_t = sum(s["total"] for s in norag_stats.values())
    norag_q = sum(s["has_quote"] for s in norag_stats.values())
    rag_ref = sum(s["has_ref"] for s in rag_stats.values())
    norag_ref = sum(s["has_ref"] for s in norag_stats.values())

    print("\n" + "-" * 72)
    print("RAG UPLIFT SUMMARY")
    print("-" * 72)
    print(f"  Evidence-quote yield:  RAG {_pct(rag_q, rag_t)}  vs  No-RAG {_pct(norag_q, norag_t)}"
          f"  (Δ = {100*(rag_q/rag_t - norag_q/norag_t):+.1f} pp)" if rag_t and norag_t else "")
    print(f"  Protocol-ref hit rate: RAG {_pct(rag_ref, rag_t)}  vs  No-RAG {_pct(norag_ref, norag_t)}"
          f"  (Δ = {100*(rag_ref/rag_t - norag_ref/norag_t):+.1f} pp)" if rag_t and norag_t else "")

    # ── Save CSV ──
    df = pd.DataFrame(rows)
    df.to_csv(PATH_METRICS_CSV, index=False)
    print(f"\n  ➜ Metrics saved to {PATH_METRICS_CSV}")

    return df


def compare_results():
    print("Starting qualitative comparison: RAG vs NO-RAG\n")
    
    no_rag_data = load_jsonl(PATH_NO_RAG)
    rag_data = load_jsonl(PATH_WITH_RAG)
    
    common_keys = set(no_rag_data.keys()).intersection(set(rag_data.keys()))
    print(f"Common cases for analysis: {len(common_keys)}")
    
    # Counters
    rag_wins_safety = 0   # RAG stayed silent, No-RAG hallucinated
    rag_wins_context = 0  # RAG brought new/different info
    identical_long = 0    # Both provided the same long text
    identical_empty = 0   # Both stayed silent (short/empty)
    rag_hallucinated = 0  # RAG spoke, but Auditor flagged it
    
    examples = []

    for key in common_keys:
        nr = no_rag_data[key]
        wr = rag_data[key]
        
        verdict_nr = nr.get("auditor_verdict", {}) or {}
        verdict_wr = wr.get("auditor_verdict", {}) or {}
        
        if not isinstance(verdict_nr, dict): verdict_nr = {}
        if not isinstance(verdict_wr, dict): verdict_wr = {}

        quote_nr = str(verdict_nr.get("evidence_quote", "")).strip()
        quote_wr = str(verdict_wr.get("evidence_quote", "")).strip()
        
        # --- EXHAUSTIVE CLASSIFICATION ---
        
        # 1. Check Warning in RAG (Hallucination caught by the system itself)
        if "Warning" in quote_wr:
            rag_hallucinated += 1
            # Does not count as a win; counts as an internal "Safety Catch"
        
        # 2. Safety Win: No-RAG hallucinated (>10 chars), RAG was honest (empty or very short)
        elif len(quote_nr) > 10 and len(quote_wr) < 5:
            rag_wins_safety += 1
            if len([e for e in examples if "SAFETY" in e['type']]) < 2:
                examples.append({
                    "type": "SAFETY WIN",
                    "id": key,
                    "condition": wr.get("cohort", "unknown"),
                    "no_rag": quote_nr[:100],
                    "rag": "[EMPTY]"
                })

        # 3. Context Win: RAG brought relevant text (>20 chars) and DIFFERENT from No-RAG
        elif len(quote_wr) > 20 and quote_wr != quote_nr:
            rag_wins_context += 1
            if len([e for e in examples if "CONTEXT" in e['type']]) < 2:
                examples.append({
                    "type": "CONTEXT WIN",
                    "id": key,
                    "condition": wr.get("cohort", "unknown"),
                    "no_rag": quote_nr[:100] if quote_nr else "[None]",
                    "rag": quote_wr[:100]
                })

        # 4. Identical Long: Both brought the SAME long quote
        elif len(quote_wr) > 20 and quote_wr == quote_nr:
            identical_long += 1
            # These are likely the 18 disappearing cases
        
        # 5. Identical Empty/Short: Both empty or short
        else:
            identical_empty += 1

    total_classified = rag_wins_safety + rag_wins_context + identical_long + identical_empty + rag_hallucinated
    
    print("\n" + "="*60)
    print("FINAL RESULTS (TOTAL VERIFIED)")
    print("="*60)
    print(f"1. Safety Wins (RAG suppressed hallucination): {rag_wins_safety}")
    print(f"2. Context Wins (RAG brought new info): {rag_wins_context}")
    print(f"3. Hallucination Catches (RAG tried, Auditor blocked): {rag_hallucinated}")
    print(f"4. Identical (Agreement on long text): {identical_long}")
    print(f"5. Identical (Both Empty): {identical_empty}")
    print("-" * 60)
    print(f"TOTAL CLASSIFIED: {total_classified} / {len(common_keys)}")
    
    print("\n" + "="*60)
    print("EXAMPLES")
    print("="*60)
    for i, ex in enumerate(examples):
        print(f"[{ex['type']}] ID: {ex['id']} ({ex['condition']})")
        print(f"   No-RAG: {ex['no_rag']}...")
        print(f"   RAG:    {ex['rag']}...")
        print("")

if __name__ == "__main__":
    # 1. Quantitative RAG validation metrics (new)
    quantitative_rag_metrics()

    # 2. Original qualitative comparison
    compare_results()