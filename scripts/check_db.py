"""Diagnostic script to inspect the ChromaDB vector store collections."""

import os

import chromadb
from chromadb.config import Settings

# 1. Define the exact path where the database should be
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
persist_directory = os.path.join(project_root, "chroma_db")

print(f"DATABASE INVESTIGATION")
print(f"Target path: {persist_directory}")

if not os.path.exists(persist_directory):
    print("CRITICAL ERROR: The 'chroma_db' folder DOES NOT EXIST at this path.")
    print("   Solution: Run the 'ingest_knowledge.py' script again.")
    exit(1)

try:
    # Connect directly to ChromaDB (without LangChain)
    client = chromadb.PersistentClient(path=persist_directory)
    
    collections = client.list_collections()
    
    if not collections:
        print("The database folder exists, but it has NO COLLECTIONS inside.")
        exit(1)
        
    print(f"\nConnection successful! Found {len(collections)} collection(s):")
    print("="*60)
    print(f"{'COLLECTION NAME':<30} | {'DOC COUNT':<15} | {'STATUS'}")
    print("-" * 60)
    
    for col in collections:
        count = col.count()
        status = "OK" if count > 0 else "EMPTY"
        print(f"{col.name:<30} | {count:<15} | {status}")
        
    print("="*60)
    
    # Analysis for the TCC
    target_col = "clinical_guidelines"
    found_target = any(c.name == target_col for c in collections)
    
    if not found_target:
        print(f"\nDIAGNOSIS: Your code 'clinical_rag.py' looks for '{target_col}',")
        print(f"   but that collection DOES NOT EXIST. Ingestion saved under another name (probably 'langchain').")
        print(f"   SOLUTION: Edit 'src/agents/clinical_rag.py' and change 'collection_name' to the name listed above.")

except Exception as e:
    print(f"Error reading the database: {e}")