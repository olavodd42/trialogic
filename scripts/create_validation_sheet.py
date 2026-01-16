import json
import csv
import os

# Ficheiros
INPUT_FILE = "results/experiment_results_v1.jsonl"
OUTPUT_FILE = "results/human_validation_sheet.csv"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Ficheiro {INPUT_FILE} não encontrado.")
        return

    print(f"📖 A ler dados da IA...")
    
    rows = []
    # Cabeçalho do Excel
    headers = [
        "Subject_ID", 
        "Cohort", 
        "IA_Vitals_Summary", 
        "IA_Risk_Score", 
        "IA_Verdict", 
        "IA_Reasoning",
        "HUMAN_AGREEMENT (Y/N)",  # <--- Preencher aqui
        "HUMAN_CORRECTION",       # <--- Se N, qual o correto?
        "NOTES"                   # <--- Obs
    ]

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Formata os vitais para caber numa célula
                vitals = data.get("extracted_vitals", {})
                vitals_str = (
                    f"HR:{vitals.get('heartrate')} | "
                    f"SBP:{vitals.get('sbp')} | "
                    f"RR:{vitals.get('resprate')} | "
                    f"Temp:{vitals.get('temperature')} | "
                    f"O2:{vitals.get('o2sat')}%"
                )

                # Formata o score
                risk_raw = data.get("risk_score", "")
                # Tenta pegar apenas o número final se possível, ou deixa o texto
                risk_str = risk_raw.replace("\n", " || ") 

                # Dados do Auditor
                audit = data.get("auditor_verdict", {})
                
                row = [
                    data.get("subject_id"),
                    data.get("cohort"),
                    vitals_str,
                    risk_str[:150], # Corta texto muito longo
                    audit.get("compliance"),
                    audit.get("evidence_quote"),
                    "", # Espaço para Humano
                    "", # Espaço para Humano
                    ""  # Espaço para Humano
                ]
                rows.append(row)
            except Exception as e:
                print(f"Erro na linha: {e}")

    # Selecionar 50 aleatórios (ou os primeiros 50 se já tiverem sido randomizados)
    # Como o batch process já processou, vamos pegar todos para você escolher.
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f: # utf-8-sig para abrir bem no Excel
        writer = csv.writer(f, delimiter=';') # Ponto e vírgula funciona melhor no Excel PT/BR
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"✅ Ficheiro criado: {OUTPUT_FILE}")
    print("👉 Abra este ficheiro no Excel.")
    print("👉 Preencha a coluna 'HUMAN_AGREEMENT (Y/N)'.")

if __name__ == "__main__":
    main()