import pandas as pd
import json
import re
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error
import logging

# Configuração de Logs para rastreabilidade (Essencial para TCC)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Captura "RESULTADO MEWS: <numero>"
MEWS_PATTERN = re.compile(r"RESULTADO MEWS:\s*(\d+)")
# Captura "RESULTADO NEWS: <numero>"
NEWS_PATTERN = re.compile(r"RESULTADO NEWS:\s*(\d+)")

def calculate_mews_python(row):
    score = 0

    if pd.isna(row['triage_sbp']) or pd.isna(row['triage_heartrate']) or \
       pd.isna(row['triage_resprate']) or pd.isna(row['triage_temperature']):
        return -1
    
    sbp = row['triage_sbp']
    if sbp <= 70: score += 3
    elif sbp <= 80: score += 2
    elif sbp <= 100: score += 1
    elif sbp >= 200: score += 2

    hr = row['triage_heartrate']
    if hr <= 40: score += 2
    elif hr <= 50: score += 1
    elif hr >= 130: score += 3
    elif hr >= 110: score += 2

    rr = row['triage_resprate']
    if rr <= 8: score += 2
    elif rr >= 30: score += 3
    elif rr >= 21: score += 2
    elif rr >= 15: score += 1

    temp = row['triage_temperature']
    if temp > 50: temp = (temp-32)*(5/9)

    if temp < 35: score += 2
    elif temp >= 38.5: score += 2

    return score

def calculate_news_python(row):
    """
    Implementação RÍGIDA do NEWS2 (Standard Scale).
    Referência: Royal College of Physicians (UK).
    
    Limitações do Dataset MIMIC-IV-ED (CSV):
    1. Assume-se 'Air' (0 pontos) para Oxigênio Suplementar se a coluna não existir.
    2. Assume-se 'Alert' (0 pontos) para AVPU se não houver GCS/AVPU estruturado.
    """
    score = 0
    
    # --- 1. Verificação de Integridade (Safety Check) ---
    # Se qualquer sinal vital crítico for NaN, não podemos calcular com confiança.
    required_cols = ['triage_resprate', 'triage_o2sat', 'triage_sbp', 'triage_heartrate', 'triage_temperature']
    for col in required_cols:
        if pd.isna(row.get(col)): # .get evita crash se a coluna não existir
            return -1 # Código de erro: Dados insuficientes

    # --- 2. Frequência Respiratória (bpm) ---
    rr = row['triage_resprate']
    if rr <= 8: score += 3
    elif rr >= 25: score += 3
    elif rr >= 21: score += 2
    elif rr >= 9 and rr <= 11: score += 1
    # 12-20 é 0

    # --- 3. Saturação de Oxigênio (%) ---
    # Assumindo Escala 1 (Padrão). Pacientes com DPOC (Scale 2) precisariam de flag específica.
    spo2 = row['triage_o2sat']
    if spo2 <= 91: score += 3
    elif spo2 >= 96: score += 0
    elif spo2 >= 94: score += 1
    elif spo2 >= 92: score += 2
    
    # --- 4. Oxigênio Suplementar ---
    # O MIMIC estruturado nem sempre tem isso claro na triagem.
    # Se você tiver a coluna, use-a. Caso contrário, assumimos 0 (False).
    uses_oxygen = row.get('supplemental_oxygen', False) 
    # Tech Lead Tip: Se o dado vier como 'True'/'False' string ou 1/0, normalize aqui.
    if uses_oxygen: 
        score += 2

    # --- 5. Pressão Arterial Sistólica (mmHg) ---
    sbp = row['triage_sbp']
    if sbp <= 90: score += 3
    elif sbp <= 100: score += 2
    elif sbp <= 110: score += 1
    # > 111 é 0. Nota: NEWS2 não pontua hipertensão isolada (diferente do MEWS).

    # --- 6. Frequência Cardíaca (bpm) ---
    hr = row['triage_heartrate']
    if hr <= 40: score += 3
    elif hr >= 131: score += 3
    elif hr >= 111: score += 2
    elif hr >= 91: score += 1
    elif hr >= 41 and hr <= 50: score += 1
    # 51-90 é 0

    # --- 7. Temperatura (°C) ---
    temp = row['triage_temperature']
    # Conversão de segurança caso esteja em Fahrenheit (> 50 provavel é F)
    if temp > 50: 
        temp = (temp - 32) * 5/9
        
    if temp <= 35.0: score += 3
    elif temp >= 39.1: score += 2
    elif temp >= 38.1: score += 1
    elif temp >= 35.1 and temp <= 36.0: score += 1
    # 36.1 - 38.0 é 0

    # --- 8. Consciência (ACVPU) ---
    # Tentativa de inferir pelo GCS se existir, senão assume Alerta
    gcs = row.get('gcs', 15) # Default 15 (Normal) se coluna faltar
    if pd.notna(gcs) and gcs < 15:
        score += 3 # Qualquer confusão ou não-alerta é 3 pontos no NEWS
        
    return score

def parse_agent_output(json_line):
    """
    Parser robusto que extrai TODOS os vitais e Scores.
    """
    data = json.loads(json_line) if isinstance(json_line, str) else json_line
    vitals = data.get("extracted_vitals", {}) or {}

    risk_text = data.get("risk_score", "")
    
    # Extração via Regex
    mews_match = MEWS_PATTERN.search(str(risk_text))
    news_match = NEWS_PATTERN.search(str(risk_text))

    system_error = "ERRO:" in str(risk_text) or "Faltam dados" in str(risk_text)
    
    return {
        "hadm_id": data.get("hadm_id"), # Chave de Junção
        "subject_id": data.get("subject_id"),
        
        # Sinais Vitais (Predictions)
        "agent_hr": vitals.get("heartrate"),
        "agent_sbp": vitals.get("sbp"),
        "agent_dbp": vitals.get("dbp"), # Adicionado DBP
        "agent_temp": vitals.get("temperature"),
        "agent_rr": vitals.get("resprate"),
        "agent_o2sat": vitals.get("o2sat"), # Adicionado O2Sat
        
        # Scores Calculados (Reasoning)
        "agent_mews": int(mews_match.group(1)) if mews_match else -1,
        "agent_news": int(news_match.group(1)) if news_match else -1,
        
        # Robustez
        "agent_reported_error": system_error
    }

DATA_PATH = os.path.join(os.getcwd(), "data/master_dataset.csv")
PREDICT_PATH = os.path.join(os.getcwd(), "results/experiment_results_v2.jsonl")

logger.info("Carregando CSV de Ground Truth...")
df_truth = pd.read_csv(DATA_PATH)

def normalize_temperature_to_celsius(val):
    if pd.isna(val): return val
    # Se for maior que 50, assumimos Fahrenheit (ninguém sobrevive a 50°C)
    if val > 50:
        return (val - 32) * 5 / 9
    return val

logger.info("Normalizando Temperatura (Fahrenheit -> Celsius)...")
df_truth['triage_temperature'] = df_truth['triage_temperature'].apply(normalize_temperature_to_celsius)

logger.info("Gerando Ground Truth para Scores (MEWS/NEWS)...")
df_truth["true_news"] = df_truth.apply(calculate_news_python, axis=1)
df_truth["true_mews"] = df_truth.apply(calculate_mews_python, axis=1)

logger.info("Carregando Predições do Agente...")
predictions = []
with open(PREDICT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue

df_preds = pd.DataFrame([parse_agent_output(p) for p in predictions])
df = pd.merge(df_truth, df_preds, on="hadm_id", how="inner")

logger.info(f"Dataset de Avaliação Consolidado: {len(df)} registros pareados.")
valid_hr = df.dropna(subset=['triage_heartrate'])
metrics_map = {
    "Heart Rate": ("triage_heartrate", "agent_hr"),
    "SBP": ("triage_sbp", "agent_sbp"),
    "Resp Rate": ("triage_resprate", "agent_rr"),
    "Temperature": ("triage_temperature", "agent_temp"),
    "O2 Sat": ("triage_o2sat", "agent_o2sat"),
    "MEWS Score": ("true_mews", "agent_mews"),
    "NEWS Score": ("true_news", "agent_news") # Descomente se usar NEWS
}

results_report = []

print("\n" + "="*60)
print(f"{'METRICA':<15} | {'ACURÁCIA (Exact)':<18} | {'MAE (Erro Médio)':<18} | {'N (Amostra)':<10}")
print("="*60)

for metric_name, (gt_col, pred_col) in metrics_map.items():
    valid_data = df.dropna(subset=[gt_col, pred_col])
    
    # Se for MEWS/NEWS, remover os -1 se quisermos avaliar apenas cálculos válidos
    if "Score" in metric_name:
        valid_data = valid_data[valid_data[gt_col] != -1]
        valid_data = valid_data[valid_data[pred_col] != -1]

    if len(valid_data) == 0:
        print(f"{metric_name:<15} | {'Sem dados':<18} | {'-':<18} | 0")
        continue

    y_true = valid_data[gt_col]
    y_pred = valid_data[pred_col]

    # 2. Métricas
    acc = (y_true.round(1) == y_pred.round(1)).mean()
    mae = mean_absolute_error(y_true, y_pred)

    # LÓGICA DE ACURÁCIA HÍBRIDA
    if metric_name == "Temperature":
        # Para temperatura, aceitamos erro de até 0.2 graus (conversão F->C gera dízimas)
        acc = (abs(y_true - y_pred) <= 0.2).mean()
    elif metric_name in ["SBP", "Heart Rate", "Resp Rate", "O2 Sat"]:
        # Para vitais numéricos, tolerância de 1 unidade é aceitável clinicamente?
        # Para rigor acadêmico, vamos manter EXACT com arredondamento, mas o MAE conta a história real.
        acc = (y_true.round(0) == y_pred.round(0)).mean()
    else:
        # Scores (MEWS/NEWS) devem ser exatos
        acc = (y_true == y_pred).mean()

    print(f"{metric_name:<15} | {acc:.2%}           | {mae:.4f}             | {len(valid_data)}")
    
    results_report.append({
        "Metric": metric_name,
        "Accuracy": acc,
        "MAE": mae,
        "N": len(valid_data)
    })

print("="*60)

# --- 4. ANÁLISE DE ROBUSTEZ (SYSTEM FAILURE) ---
# Quantas vezes o Python disse "Não dá pra calcular" (-1) e o Agente concordou?
df_errors = df[df["true_mews"] == -1]
if len(df_errors) > 0:
    correct_rejections = df_errors["agent_reported_error"].sum()
    rejection_rate = correct_rejections / len(df_errors)
    print(f"\n[Robustez] Taxa de Rejeição Correta (Dados Faltantes): {rejection_rate:.2%} ({correct_rejections}/{len(df_errors)})")
else:
    print("\n[Robustez] Não houve casos de dados faltantes no Ground Truth para avaliar rejeição.")