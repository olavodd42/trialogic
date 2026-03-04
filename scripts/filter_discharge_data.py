"""Filter discharge.csv to keep only rows matching master_dataset hadm_ids."""

import os

import pandas as pd


def filter_discharge_data():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    master_path = os.path.join(base_path, 'master_dataset.csv')
    discharge_path = os.path.join(base_path, 'discharge.csv')
    output_path = os.path.join(base_path, 'discharge_filtered.csv')

    print(f"Reading {master_path}...")
    # Read only the hadm_id column to save memory
    df_master = pd.read_csv(master_path, usecols=['hadm_id'])
    
    # Filter valid (non-null) hadm_ids and convert to int
    valid_hadm_ids = df_master['hadm_id'].dropna().astype(int).unique()
    print(f"Found {len(valid_hadm_ids)} unique hadm_ids in master_dataset.csv")

    print(f"Reading {discharge_path}...")
    df_discharge = pd.read_csv(discharge_path)
    
    print(f"Total records in discharge.csv: {len(df_discharge)}")

    # Ensure hadm_id in discharge is also handled correctly
    # There may be nulls or mixed types, so force numeric and dropna
    df_discharge['hadm_id'] = pd.to_numeric(df_discharge['hadm_id'], errors='coerce')
    df_discharge = df_discharge.dropna(subset=['hadm_id'])
    df_discharge['hadm_id'] = df_discharge['hadm_id'].astype(int)

    # Filter
    df_filtered = df_discharge[df_discharge['hadm_id'].isin(valid_hadm_ids)]
    
    print(f"Records after filtering: {len(df_filtered)}")

    print(f"Saving to {output_path}...")
    df_filtered.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    filter_discharge_data()
