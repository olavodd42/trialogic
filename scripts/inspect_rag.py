import json
import os
from collections import Counter

FILE_PATH = "results/experiment_results_v1.jsonl"

def analyze_file():
    print(f"🕵️‍♂️  ANALISANDO O ARQUIVO: {FILE_PATH}\n")
    
    if not os.path.exists(FILE_PATH):
        print("❌ Arquivo não encontrado.")
        return

    total_lines = 0
    success_count = 0
    error_count = 0
    
    error_types = Counter()
    cohorts = Counter()
    
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                
                # Verifica se é erro
                if "error" in data:
                    error_count += 1
                    # Pega o começo do erro para agrupar
                    err_msg = str(data['error'])
                    if "Recursion limit" in err_msg:
                        error_types["Recursion Limit (Loop Infinito)"] += 1
                    elif "OutputParserException" in err_msg:
                        error_types["Parser Error (JSON Inválido)"] += 1
                    else:
                        error_types[err_msg[:50]] += 1
                else:
                    success_count += 1
                    cohorts[data.get('cohort', 'unknown')] += 1
                    
            except json.JSONDecodeError:
                print(f"⚠️ Linha {total_lines} corrompida (não é JSON válido).")

    print("📊 ESTATÍSTICAS GERAIS:")
    print(f"   - Total de Linhas: {total_lines}")
    print(f"   - Sucessos (Processados): {success_count} ({(success_count/total_lines)*100:.1f}%)")
    print(f"   - Erros (Falhas): {error_count} ({(error_count/total_lines)*100:.1f}%)")
    
    if error_count > 0:
        print("\n❌ ANÁLISE DE ERROS:")
        for err, count in error_types.items():
            print(f"   - {count}x: {err}...")

    print("\n✅ DISTRIBUIÇÃO DOS SUCESSOS (COHORTS):")
    for cohort, count in cohorts.items():
        print(f"   - {cohort}: {count}")

if __name__ == "__main__":
    analyze_file()