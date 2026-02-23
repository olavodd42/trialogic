import logging
import os
import json
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# 1. Configuration
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    seed=42
)

# 2. One-shot Prompt
PROMPT_PATH = "prompts/one_shot_prompt.md"
try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        BASELINE_PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    logger.error(f"CRITICAL: Prompt file not found at {PROMPT_PATH}")
    exit(1)


def robust_json_extractor(text: str) -> Optional[Dict]:
    import re, json
    # Captures all blocks between first `{` and last `}`
    try:
        # 1. Try direct clean extraction
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Removes markdown blocks (```json ... ```)
    clean_text = re.sub(r'```json\s*', '', text)
    clean_text = re.sub(r'```\s*', '', clean_text)
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # 3. Finds the biggest valid JSON object in the string (Greedy Match)
    match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    return None

def process_baseline_batch(df: pd.DataFrame, limit: Optional[int] = None):
    results = []
    
    # 3. Limit for fast test if necessary
    if limit:
        df = df.head(limit)
        
    logger.info(f"🚀 Starting baseline (Vanilla Llama 3.1) in {len(df)} cases...")
    iterator = tqdm(df.iterrows(), total=df.shape[0], desc="Processing Clinical Notes")
    for index, row in iterator:
        text = str(row.get('text', ''))
        hadm_id = row.get('hadm_id', index)

        if pd.isna(text) or str(text).strip() == "":
                results.append({"hadm_id": hadm_id, "error": "EMPTY_TEXT"})
                continue
        
        try:
            # 3. Zero-shot inference (dummy)
            prompt = BASELINE_PROMPT_TEMPLATE.replace("{clinical_text}", str(text))
            # prompt = BASELINE_PROMPT.format(clinical_text=text)
            response = llm.invoke([
                HumanMessage(content=prompt)
            ])
            
            raw_content = response.content

            # Ensure raw_content is a string
            raw_content_str = str(raw_content)

            logger.debug(raw_content_str)
            # 4. Try to parse the response
            parsed = robust_json_extractor(raw_content_str)
            
            if parsed:
                result_entry = {
                    "hadm_id": row.get('hadm_id'), # CHAVE DE JOIN
                    "subject_id": row.get('subject_id'),
                    "cohort": row.get('cohort_type', 'sepsis'),
                    "extracted_vitals": parsed.get("extracted_vitals", {}),
                    "risk_score": parsed.get("text_report", ""),
                    "method": "baseline_zero_shot",
                    "raw_response": raw_content,
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

    return results

if __name__ == "__main__":
    # Caminhos
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use data/gold_standard_dataset.csv for fair comparison with run_batch_processing.py
    CSV_PATH = os.path.join(BASE_DIR, "data/gold_standard_dataset.csv") 
    OUTPUT_PATH = os.path.join(BASE_DIR, "results/oneshot_baseline_results.jsonl")
    
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
