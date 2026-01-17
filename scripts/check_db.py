import os
import chromadb
from chromadb.config import Settings

# 1. Definir o caminho exato onde o banco deveria estar
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
persist_directory = os.path.join(project_root, "chroma_db")

print(f"🕵️  INVESTIGAÇÃO DO BANCO DE DADOS")
print(f"📂 Caminho Alvo: {persist_directory}")

if not os.path.exists(persist_directory):
    print("❌ ERRO CRÍTICO: A pasta 'chroma_db' NÃO EXISTE neste caminho.")
    print("   Solução: Rode o script 'ingest_knowledge.py' novamente.")
    exit(1)

try:
    # Conecta direto no ChromaDB (sem passar pelo LangChain)
    client = chromadb.PersistentClient(path=persist_directory)
    
    collections = client.list_collections()
    
    if not collections:
        print("⚠️  O banco existe (pasta), mas NÃO TEM NENHUMA COLEÇÃO dentro.")
        exit(1)
        
    print(f"\n✅ Conexão bem sucedida! Encontrei {len(collections)} coleção(ões):")
    print("="*60)
    print(f"{'NOME DA COLEÇÃO':<30} | {'QTD DOCUMENTOS':<15} | {'STATUS'}")
    print("-" * 60)
    
    for col in collections:
        count = col.count()
        status = "🟢 CHEIA" if count > 0 else "🔴 VAZIA"
        print(f"{col.name:<30} | {count:<15} | {status}")
        
    print("="*60)
    
    # Análise para o TCC
    target_col = "clinical_guidelines"
    found_target = any(c.name == target_col for c in collections)
    
    if not found_target:
        print(f"\n🚨 DIAGNÓSTICO: O seu código 'clinical_rag.py' procura por '{target_col}',")
        print(f"   mas essa coleção NÃO EXISTE. O ingest salvou com outro nome (provavelmente 'langchain').")
        print(f"   👉 SOLUÇÃO: Edite 'src/agents/clinical_rag.py' e mude 'collection_name' para o nome listado acima.")

except Exception as e:
    print(f"❌ Erro ao ler o banco: {e}")