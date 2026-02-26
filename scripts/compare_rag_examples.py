import json

ta = {}
with open("results/experiment_results_v1.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        hid = r.get("hadm_id")
        if hid and "auditor_verdict" in r:
            ta[hid] = r

nr = {}
with open("results/norag_experiment_results_v1.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        hid = r.get("hadm_id")
        if hid and "auditor_verdict" in r:
            nr[hid] = r

common = sorted(set(ta.keys()) & set(nr.keys()))

# Find best examples: TA has real quote+reference, NR has empty/generic
best = []
for hid in common:
    tv = ta[hid]["auditor_verdict"]
    nv = nr[hid]["auditor_verdict"]
    ta_q = tv.get("evidence_quote", "")
    nr_q = nv.get("evidence_quote", "")
    ta_ref = tv.get("protocol_reference", "")
    ta_sug = tv.get("clinical_suggestion", "")
    nr_sug = nv.get("clinical_suggestion", "")
    # TA has substantial quote and NR does not
    if ta_ref and len(ta_q) > 30 and len(nr_q) < 15 and ta_sug != nr_sug:
        best.append(hid)

print(f"Total best examples: {len(best)}")
print()

# Show top 5
for hid in best[:5]:
    tv = ta[hid]["auditor_verdict"]
    nv = nr[hid]["auditor_verdict"]
    cohort = ta[hid].get("cohort", "?")
    print("=" * 80)
    print(f"HADM_ID: {hid} | Cohort: {cohort}")
    print()
    print("--- TA (com RAG) ---")
    print(f"  protocol_reference: {tv.get('protocol_reference', '')}")
    print(f"  risk_category:      {tv.get('clinical_risk_category', '')}")
    print(f"  evidence_quote:     {tv.get('evidence_quote', '')[:300]}")
    print(f"  suggestion:         {tv.get('clinical_suggestion', '')}")
    print(f"  reasoning:          {tv.get('reasoning_trace', '')[:300]}")
    print()
    print("--- NR (sem RAG) ---")
    print(f"  protocol_reference: {nv.get('protocol_reference', '')}")
    print(f"  risk_category:      {nv.get('clinical_risk_category', '')}")
    print(f"  evidence_quote:     {nv.get('evidence_quote', '')[:300]}")
    print(f"  suggestion:         {nv.get('clinical_suggestion', '')}")
    print(f"  reasoning:          {nv.get('reasoning_trace', '')[:300]}")
    print()
