import json
import os
import pandas as pd
from typing import Dict, Any

# --- CONFIGURAÇÃO DE CAMINHOS ---
PATH_NO_RAG = "results/norag_experiment_results_v1.jsonl" 
PATH_WITH_RAG = "results/experiment_results_v1.jsonl" 

def load_jsonl(path: str) -> Dict[str, Any]:
    data = {}
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
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

def compare_results():
    print("🕵️‍♂️  Starting Qualitative Comparison: RAG vs NO-RAG (Logic Fixed)\n")
    
    no_rag_data = load_jsonl(PATH_NO_RAG)
    rag_data = load_jsonl(PATH_WITH_RAG)
    
    common_keys = set(no_rag_data.keys()).intersection(set(rag_data.keys()))
    print(f"📊 Common cases for analysis: {len(common_keys)}")
    
    # Contadores
    rag_wins_safety = 0   # RAG ficou quieto, No-RAG falou
    rag_wins_context = 0  # RAG trouxe algo novo/diferente
    identical_long = 0    # Ambos falaram a mesma coisa (Longa)
    identical_empty = 0   # Ambos ficaram quietos (Curta/Vazia)
    rag_hallucinated = 0  # RAG falou, mas Auditor avisou que é mentira
    
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
        
        # --- CLASSIFICAÇÃO EXAUSTIVA ---
        
        # 1. Checar Warning no RAG (Alucinação detectada pelo próprio sistema)
        if "Warning" in quote_wr:
            rag_hallucinated += 1
            # Não conta como vitória, conta como "Safety Catch" interno
        
        # 2. Safety Win: No-RAG alucinou (>10 chars), RAG foi honesto (Vazio ou muito curto)
        elif len(quote_nr) > 10 and len(quote_wr) < 5:
            rag_wins_safety += 1
            if len([e for e in examples if "SAFETY" in e['type']]) < 2:
                examples.append({
                    "type": "🛡️ SAFETY WIN",
                    "id": key,
                    "condition": wr.get("cohort", "unknown"),
                    "no_rag": quote_nr[:100],
                    "rag": "[EMPTY]"
                })

        # 3. Context Win: RAG trouxe texto relevante (>20 chars) e DIFERENTE do No-RAG
        elif len(quote_wr) > 20 and quote_wr != quote_nr:
            rag_wins_context += 1
            if len([e for e in examples if "CONTEXT" in e['type']]) < 2:
                examples.append({
                    "type": "📚 CONTEXT WIN",
                    "id": key,
                    "condition": wr.get("cohort", "unknown"),
                    "no_rag": quote_nr[:100] if quote_nr else "[None]",
                    "rag": quote_wr[:100]
                })

        # 4. Identical Long: Ambos trouxeram a MESMA citação longa
        elif len(quote_wr) > 20 and quote_wr == quote_nr:
            identical_long += 1
            # Isso provavelmente são os 18 casos sumidos
        
        # 5. Identical Empty/Short: Ambos vazios ou curtos
        else:
            identical_empty += 1

    total_classified = rag_wins_safety + rag_wins_context + identical_long + identical_empty + rag_hallucinated
    
    print("\n" + "="*60)
    print("🚀  FINAL RESULTS (TOTAL VERIFIED)")
    print("="*60)
    print(f"1. Safety Wins (RAG suppressed hallucination): {rag_wins_safety}")
    print(f"2. Context Wins (RAG brought new info): {rag_wins_context}")
    print(f"3. Hallucination Catches (RAG tried, Auditor blocked): {rag_hallucinated}")
    print(f"4. Identical (Agreement on long text): {identical_long}")
    print(f"5. Identical (Both Empty): {identical_empty}")
    print("-" * 60)
    print(f"✅ TOTAL CLASSIFIED: {total_classified} / {len(common_keys)}")
    
    print("\n" + "="*60)
    print("💡  EXAMPLES")
    print("="*60)
    for i, ex in enumerate(examples):
        print(f"[{ex['type']}] ID: {ex['id']} ({ex['condition']})")
        print(f"   🔴 No-RAG: {ex['no_rag']}...")
        print(f"   🟢 RAG:    {ex['rag']}...")
        print("")

if __name__ == "__main__":
    compare_results()