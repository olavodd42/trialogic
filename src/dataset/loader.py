import os
import pandas as pd
import polars as pl

def load_discharge_dataset(file_path: str | None = None) -> pl.DataFrame:
    """
    Carrega o dataset de discharge.csv usando Polars para alta performance.
    Retorna um DataFrame do Polars.
    """
    if file_path is None:
        # Define o caminho padrão relativo à raiz do projeto
        # Estrutura esperada: root/src/dataset/loader.py -> root/data/discharge.csv
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        file_path = os.path.join(project_root, "data", "discharge.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    print(f"Carregando dataset de: {file_path}...")
    
    try:
        # Polars é otimizado para leitura rápida de arquivos grandes
        df = pl.read_csv(file_path)
        print(f"Dataset carregado com sucesso! Dimensões: {df.shape}")
        return df
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        raise e

if __name__ == "__main__":

    df = load_discharge_dataset()
    print(df.head())