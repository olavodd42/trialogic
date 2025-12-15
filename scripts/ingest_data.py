# scripts/ingest_data.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 1. Carregar variáveis de ambiente
load_dotenv()

# 2. Configurar conexão
DB_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@" \
         f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

engine = create_engine(DB_URL)

CSV_PATH = "data/master_dataset.csv" # Ajuste o caminho conforme necessário
TABLE_NAME = "ed_triage"

def ingest_data() -> None:
    print(f"🚀 Iniciando ingestão de {CSV_PATH} para a tabela '{TABLE_NAME}'...")
    
    # 3. Ler em chunks (lotes) para não estourar a memória
    chunk_size = 10000
    total_rows = 0
    
    try:
        # Cria um iterador
        with pd.read_csv(CSV_PATH, chunksize=chunk_size) as reader:
            for i, chunk in enumerate(reader):
                # Limpeza básica se necessário (ex: converter datas)
                if 'charttime' in chunk.columns:
                    chunk['charttime'] = pd.to_datetime(chunk['charttime'])
                
                # Inserir no banco
                mode = 'replace' if i == 0 else 'append'
                
                chunk.to_sql(
                    TABLE_NAME, 
                    engine, 
                    if_exists=mode, 
                    index=False,
                    method='multi' # Otimização de inserção
                )
                
                total_rows += len(chunk)
                print(f"📦 Lote {i+1} processado. Total de linhas: {total_rows}")
        
        print("✅ Ingestão concluída com sucesso!")
        
        # 4. Criar Índices (CRUCIAL PARA PERFORMANCE DO AGENTE)
        create_indexes()

    except Exception as e:
        print(f"❌ Erro fatal: {e}")

def create_indexes() -> None:
    """
    Cria índices para buscas rápidas pelo Agente.
    Sem isso, o RAG vai demorar segundos para achar um paciente.
    """
    print("⚡ Criando índices de performance...")
    with engine.connect() as con:
        # Supondo que 'stay_id' ou 'subject_id' sejam chaves primárias lógicas no dataset de Xie
        con.execute(text(f"CREATE INDEX IF NOT EXISTS idx_stay_id ON {TABLE_NAME} (stay_id);"))
        con.execute(text(f"CREATE INDEX IF NOT EXISTS idx_subject_id ON {TABLE_NAME} (subject_id);"))
        con.execute(text(f"CREATE INDEX IF NOT EXISTS idx_hadm_id ON {TABLE_NAME} (hadm_id);"))
        con.commit()
    print("⚡ Índices criados.")

if __name__ == "__main__":
    ingest_data()