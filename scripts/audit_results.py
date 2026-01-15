import json
import os

FILE_PATH = "results/experiment_results_v1.jsonl"

def audit_the_auditor():
    if not os.path.exists(FILE_PATH):
        print("Arquivo não encontrado.")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Dados do Paciente
            subject_id = data.get("subject_id")
            vitals = data.get("extracted_vitals", {}) or {}
            sbp = vitals.get("sbp", "N/A")
            
            # Veredito da IA
            audit = data.get("auditor_verdict", {}) or {}
            verdict = audit.get("compliance", "N/A")
            evidence = audit.get("evidence_quote", "N/A")
            
            print(f"--- ID: {subject_id} ---")
            print(f"🏥 Paciente SBP: {sbp}")
            print(f"📜 Regra Citada: {evidence}")
            print(f"🤖 Veredito IA:  {verdict}")
            
            # Validação Lógica Simples (Tech Lead Check)
            try:
                if sbp != "N/A" and "90" in evidence and verdict == "Non-Compliant":
                    if float(sbp) > 90:
                        print("❌ ERRO LÓGICO DETECTADO: Paciente estável (SBP>90) marcado como risco!")
                    else:
                        print("✅ ACERTO: Paciente realmente hipotenso.")
            except:
                pass
            print("-" * 30)

if __name__ == "__main__":
    audit_the_auditor()