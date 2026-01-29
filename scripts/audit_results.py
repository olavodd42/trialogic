import pandas as pd
import json
import os

# CONFIGURAÇÃO
CSV_PATH = "data/gold_standard_dataset.csv"
JSONL_PATH = "results/experiment_results_v1.jsonl"

def audit_discrepancy():
    print("🕵️‍♂️  INICIANDO AUDITORIA DE INTEGRIDADE DE DADOS...\n")

    # 1. Carregar Ground Truth
    if not os.path.exists(CSV_PATH):
        print(f"❌ Erro: CSV não encontrado em {CSV_PATH}")
        return
    df_gt = pd.read_csv(CSV_PATH)
    total_gt = len(df_gt)
    # Normalizar IDs para string para garantir match
    gt_ids = set(df_gt['subject_id'].astype(str) + "_" + df_gt['hadm_id'].astype(str))
    
    print(f"📊 Total de Casos no Gabarito (CSV): {total_gt}")

    # 2. Carregar Resultados do Experimento
    if not os.path.exists(JSONL_PATH):
        print(f"❌ Erro: JSONL não encontrado em {JSONL_PATH}")
        return
    
    processed_rows = []
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Verifica se é registro de erro explícito
                if "error" in data:
                    processed_rows.append({"type": "error", "id": f"{data.get('subject_id')}_{data.get('hadm_id')}"})
                else:
                    processed_rows.append({"type": "success", "id": f"{data.get('subject_id')}_{data.get('hadm_id')}"})
            except:
                pass

    total_processed = len(processed_rows)
    success_ids = {row['id'] for row in processed_rows if row['type'] == 'success'}
    error_ids = {row['id'] for row in processed_rows if row['type'] == 'error'}
    
    print(f"📊 Total de Casos no Resultado (JSONL): {total_processed}")
    print(f"   ✅ Sucessos (Vitals extraídos): {len(success_ids)}")
    print(f"   ❌ Erros Explícitos (Exceptions): {len(error_ids)}")

    # 3. Análise de GAP (Quem sumiu?)
    processed_ids = success_ids.union(error_ids)
    missing_ids = gt_ids - processed_ids
    
    print("\n" + "="*40)
    print(f"📉 REGISTROS PERDIDOS (SKIPPED): {len(missing_ids)}")
    print("="*40)

    if len(missing_ids) > 0:
        print("\n🔍 Investigando causa dos perdidos (Amostra):")
        # Vamos olhar no CSV original para ver se eram textos curtos
        df_gt['unique_id'] = df_gt['subject_id'].astype(str) + "_" + df_gt['hadm_id'].astype(str)
        missing_df = df_gt[df_gt['unique_id'].isin(missing_ids)]
        
        short_text_count = 0
        for _, row in missing_df.iterrows():
            text_len = len(str(row['text']))
            if text_len < 50:
                short_text_count += 1
            
            # Printar os 3 primeiros para exemplo
            if short_text_count <= 3:
                print(f"   - ID {row['subject_id']}: Tamanho do texto = {text_len} chars")

        print(f"\n💡 Diagnóstico: {short_text_count} de {len(missing_ids)} perdidos tinham texto < 50 caracteres.")
        
        real_success_rate = len(success_ids) / total_gt
        adjusted_success_rate = len(success_ids) / (total_gt - short_text_count)
        
        print("\n" + "="*40)
        print("📈 ESTATÍSTICAS REAIS:")
        print(f"   - Taxa Bruta (vs Total CSV): {real_success_rate:.2%}")
        print(f"   - Taxa Ajustada (vs Textos Válidos): {adjusted_success_rate:.2%}")
        print("="*40)

if __name__ == "__main__":
    audit_discrepancy()