# src/utils/database.py
import os
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_db_connection() -> SQLDatabase:
    """
    Estabelece a conexão com o Postgres via SQLAlchemy e 
    envolve-a no wrapper do LangChain.
    """
    db_uri = f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@" \
             f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    engine = create_engine(db_uri)
    db = SQLDatabase(engine, include_tables=["ed_triage"])
    
    return db

# Teste rápido (Sanity Check)
if __name__ == "__main__":
    db = get_db_connection()
    print(f"✅ Conexão estabelecida. Tabelas visíveis: {db.get_usable_table_names()}")
    print(db.run("SELECT count(*) FROM ed_triage"))