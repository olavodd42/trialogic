import logging
import os
import json
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm

# 1. Configuration
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.1",
    temperature=0,
    seed=42,
    num_predict=512
)

# 2. Monolythic Prompt
BASELINE_PROMPT = """
You are a clinical AI. Your task is to Extract vitals and Calculate risk scores.

INPUT TEXT:
{clinical_text}

INSTRUCTIONS:
1. Extract: HR, SBP, DBP, RR, O2Sat, Temp, AVPU, Supplemental Oxygen.
2. [CRITICAL] Normalize: For temperature, if it is in Fahrenheit, convert to Celsius.
3. Calculate:
   - NEWS2 Score (National Early Warning Score 2)
   - MEWS Score (Modified Early Warning Score)
   *Note: If mental status is confused/disoriented, count it as AVPU score > 0.*

OUTPUT FORMAT:
Return ONLY a valid JSON object. No intro text. No markdown formatting like ```json.
DO NOT ADD COMMENTS (like //) inside the JSON.

{{
  "text_report": "SCORE TOTAL NEWS: <insert number> SCORE TOTAL MEWS: <insert number>",
  "extracted_vitals": {{
    "heartrate": <int or null>,
    "resprate": <int or null>,
    "temperature": <float or null>,
    "o2sat": <int or null>,
    "sbp": <int or null>,
    "dbp": <int or null>,
    "avpu": "<Alert|Voice|Pain|Unresponsive|Confusion>",
    "supplemental_oxygen": <true|false>
  }}
}}
"""

def robust_json_extractor(text: str) -> Optional[Dict]:
    text = re.sub(r'//.*', '', text)
    try:
        # 1. Try parsing
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            # Remove quebras de linha que podem quebrar strings
            json_str = re.sub(r'\n', ' ', json_str)
            return json.loads(json_str)
    except:
        pass

    # 2. FALLBACK
    try:
        clean = text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except:
        pass
    
    return None

def process_baseline_batch(df: pd.DataFrame, limit: Optional[int] = None):
    import time
    results = []
    latencies = []
    
    # 3. Limit for fast test if necessary
    if limit:
        df = df.head(limit)
        
    logger.info(f"🚀 Starting baseline (Vanilla Llama 3.1) in {len(df)} cases...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        start_time = time.time()
        text = str(row.get('text', ''))

        if len(text) < 15:
            continue
        
        try:
            # 3. Zero-shot inference (dummy)
            prompt = BASELINE_PROMPT.format(clinical_text=text)
            response = llm.invoke([
                HumanMessage(content=prompt)
            ])
            
            raw_content = response.content

            logger.debug(raw_content)

            # 4. Try to parse the response
            parsed = robust_json_extractor(raw_content)
            
            if parsed:
                result_entry = {
                    "hadm_id": row.get('hadm_id'), # CHAVE DE JOIN
                    "subject_id": row.get('subject_id'),
                    "cohort": row.get('cohort_type', 'sepsis'),
                    "extracted_vitals": parsed.get("extracted_vitals", {}),
                    "risk_score": parsed.get("text_report", ""),
                    "method": "baseline_zero_shot"
                }
                results.append(result_entry)
            else:
                logger.warning(f"Failed to parse JSON for ID {row.get('subject_id')}")
                results.append({
                    "hadm_id": row.get('hadm_id'),
                    "error": "JSON_PARSE_ERROR",
                    "raw_output": raw_content[:50]
                })

        except Exception as e:
            logger.error(f"Error on index {index}: {e}")
            results.append({
                "hadm_id": row.get('hadm_id'),
                "error": f"EXCEPTION: {str(e)}",
                "method": "baseline_error"
            })
        finally:
            end_time = time.time()
            latencies.append(end_time - start_time)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    logger.info(f"Tempo médio de inferência por caso (baseline zero-shot): {avg_latency:.2f} segundos")
    return results

if __name__ == "__main__":
    # Caminhos
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use data/gold_standard_dataset.csv for fair comparison with run_batch_processing.py
    CSV_PATH = os.path.join(BASE_DIR, "data/gold_standard_dataset.csv") 
    OUTPUT_PATH = os.path.join(BASE_DIR, "results/baseline_results.jsonl")
    
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # df = df.head(2) 
        if 'hadm_id' in df.columns:
            df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce')
        
        results = process_baseline_batch(df, limit=None)
        
        # Save
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
                
        logger.info(f"✅ Baseline finished. Results saved in {OUTPUT_PATH}")
    else:
        logger.error(f"Dataset not found: {CSV_PATH}")
