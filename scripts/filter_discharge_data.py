import pandas as pd
import os

def filter_discharge_data():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    master_path = os.path.join(base_path, 'master_dataset.csv')
    discharge_path = os.path.join(base_path, 'discharge.csv')
    output_path = os.path.join(base_path, 'discharge_filtered.csv')

    print(f"Lendo {master_path}...")
    # Ler apenas a coluna hadm_id para economizar memória
    df_master = pd.read_csv(master_path, usecols=['hadm_id'])
    
    # Filtrar hadm_ids válidos (não nulos) e converter para inteiro
    valid_hadm_ids = df_master['hadm_id'].dropna().astype(int).unique()
    print(f"Encontrados {len(valid_hadm_ids)} hadm_ids únicos em master_dataset.csv")

    print(f"Lendo {discharge_path}...")
    df_discharge = pd.read_csv(discharge_path)
    
    print(f"Total de registros em discharge.csv: {len(df_discharge)}")

    # Garantir que hadm_id em discharge também seja tratado corretamente
    # Pode haver nulos ou tipos mistos, então vamos forçar numérico e dropna
    df_discharge['hadm_id'] = pd.to_numeric(df_discharge['hadm_id'], errors='coerce')
    df_discharge = df_discharge.dropna(subset=['hadm_id'])
    df_discharge['hadm_id'] = df_discharge['hadm_id'].astype(int)

    # Filtrar
    df_filtered = df_discharge[df_discharge['hadm_id'].isin(valid_hadm_ids)]
    
    print(f"Registros após filtragem: {len(df_filtered)}")

    print(f"Salvando em {output_path}...")
    df_filtered.to_csv(output_path, index=False)
    print("Concluído.")

if __name__ == "__main__":
    filter_discharge_data()
