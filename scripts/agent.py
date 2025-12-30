import json
import polars as pl
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

# Importamos a versão refatorada (Nó)
from src.agents.scribe import scribe_node
from src.schemas.input_schema import InputSchema

# Configuração de Caminhos
DATA_PATH = Path("data/discharge_filtered.csv")
OUTPUT_PATH = Path("data/scribe_results.jsonl")
MAX_WORKERS = 10
N = 100

# Lock para escrita segura em arquivo
file_lock = threading.Lock()

def get_processed_ids():
    """Lê o arquivo de saída e retorna um set com os hadm_ids já processados."""
    processed_ids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "hadm_id" in record:
                        processed_ids.add(record["hadm_id"])
                except json.JSONDecodeError:
                    continue
    return processed_ids

def process_record(row):
    """Processa um único registro de forma isolada."""
    # 1. Preparar o Estado (Mock do LangGraph)
    state_mock = {
        "subject_id": row["subject_id"],
        "hadm_id": row["hadm_id"],
        "raw_text": row["text"]
    }
    
    # 2. Invocar o Nó
    result_update = scribe_node(state_mock)
    
    # 3. Processar o Resultado
    if result_update.get("scribe_status") == "success":
        structured_data = result_update["structured_output"]
    else:
        structured_data = None

    # 4. Montar o Objeto Final
    final_record = {
        "subject_id": row["subject_id"],
        "hadm_id": row["hadm_id"],
        "structured_output": structured_data,
        "status": result_update.get("scribe_status", "unknown"),
        "error_msg": result_update.get("error_msg")
    }
    return final_record

def run_batch_processing():
    print(f"--- [BATCH RUNNER] Lendo dados de {DATA_PATH} ---")
    
    try:
        df = pl.read_csv(DATA_PATH, n_rows=N)
    except FileNotFoundError:
        print(f"Erro: Arquivo {DATA_PATH} não encontrado.")
        return

    # Identificar já processados para Resume
    print("--- Verificando registros já processados... ---")
    processed_ids = get_processed_ids()
    print(f"--- Encontrados {len(processed_ids)} registros já processados. ---")

    # Filtrar DataFrame
    rows_to_process = []
    for row in df.iter_rows(named=True):
        if row["hadm_id"] not in processed_ids:
            rows_to_process.append(row)
            
    total_to_process = len(rows_to_process)
    print(f"--- Iniciando processamento de {total_to_process} novos registros (de {len(df)} totais) com {MAX_WORKERS} workers ---")
    
    if total_to_process == 0:
        print("Todos os registros já foram processados.")
        return

    # Abre arquivo em modo 'append' (a)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f_out:
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submete todas as tarefas
            futures = [executor.submit(process_record, row) for row in rows_to_process]
            
            # Processa conforme completam
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                
                # 5. Salvar imediatamente (JSONL) com thread safety
                with file_lock:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"--- Processamento concluído. Resultados salvos em {OUTPUT_PATH} ---")

if __name__ == "__main__":
    run_batch_processing()