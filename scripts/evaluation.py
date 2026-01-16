from dataclasses import dataclass
from typing import Set, Dict, List, Any
import logging
import os
import re
import polars as pl
import pandas as pd

# Configuração de Logs para rastreabilidade (Essencial para TCC)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Captura "RESULTADO MEWS: <numero>"
MEWS_PATTERN = re.compile(r"RESULTADO MEWS:\s*(\d+)")
# Captura "RESULTADO NEWS: <numero>"
NEWS_PATTERN = re.compile(r"RESULTADO NEWS:\s*(\d+)")

@dataclass
class EvaluationMetrics:
    """Value Object para armazenar métricas de uma execução."""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int

class AgentEvaluator:
    """
    Responsável por comparar extrações clínicas (Prediction) contra Gabarito (Ground Truth).
    Segue o princípio SRP (Single Responsibility Principle).
    """

    def _normalize_entity(self, value: Any) -> str:
        """Normaliza strings para comparação justa (caixa baixa, strip, etc)."""
        return str(value).lower().strip()
    
    def parse_prediction(self, line: Dict[str, Any]) -> Dict[str, Any]:
        raw_score = line.get("risk_score", "")
        vitals = line.get("extracted_vitals", {}) or {} # Handle null

        # 1. Try to extract risk scores
        mews_match = MEWS_PATTERN.search(str(raw_score))
        news_match = NEWS_PATTERN.search(str(raw_score))
        
        # 2. Detect if it had an execution error
        has_error = "ERRO:" in str(raw_score) or "Faltam dados" in str(raw_score)

        parsed_data = {
            "subject_id": line.get("subject_id"),
            # Extraction Metrics
            "pred_temp": vitals.get("temperature"),
            "pred_hr": vitals.get("heartrate"),
            "pred_sbp": vitals.get("sbp"),
            "pred_dbp": vitals.get("dbp"),
            "pred_resp": vitals.get("resprate"),
            "pred_o2sat": vitals.get("o2sat"),
            "pred_gcs": vitals.get("gcs"),
            "pred_acuity": vitals.get("acuity"),

            # Calculated metrics
            "pred_mews": int(mews_match.group(1)) if mews_match else None,
            "pred_news": int(news_match.group(1)) if news_match else None,
            
            # Robustness metrics
            "system_error": has_error
        }

        return parsed_data

    def evaluate_batch(self, predictions: list, ground_truth_df: pd.DataFrame) -> EvaluationMetrics:
        results = []
        
        for pred_raw in predictions:
            pred = self.parse_prediction(pred_raw)
            sid = pred["subject_id"]

            ground_truth = ground_truth_df[ground_truth_df['subject_id'] == sid]
            true_temp = ground_truth["temperature"]

            

MASTER_DATASET = os.path.join(os.getcwd(), "data/master_dataset.csv")
RESULTS_EXPERIMENT = os.path.join(os.getcwd(), "results/experiment_results_v1.jsonl")

# --- Exemplo de Uso (Test Drive) ---
if __name__ == "__main__":
    master_df = pl.read_csv(MASTER_DATASET)
    predictions_dataset = pd.read_json(RESULTS_EXPERIMENT, lines=True)

    subject_ids_in_predictions = predictions_dataset["subject_id"].unique().tolist()
    gt_df = master_df.filter(pl.col("subject_id").is_in(subject_ids_in_predictions)).to_pandas()

    for _,prediction in predictions_dataset.iterrows():
        ground_truth = gt_df[gt_df["subject_id"] == prediction["subject_id"]][0]
        predicted_vitals = prediction["extracted_vitals"]
        for vital
        
    evaluator = ClinicalExtractorEvaluator()
    metrics = evaluator.calculate_ner_metrics(prediction, ground_truth)
    
    logger.info(f"Relatório de Avaliação:\n{metrics}")
    
    # Tech Lead Tip: Note que 'Dispineia' (errado) vs 'Dispneia' (certo) contou como erro.
    # Solução futura: Usar métricas de similaridade de string (Levenshtein) ou embeddings.