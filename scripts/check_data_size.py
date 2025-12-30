import polars as pl
from pathlib import Path

DATA_PATH = Path("data/discharge_filtered.csv")

def check_size():
    # Read only first 100 rows
    df = pl.read_csv(DATA_PATH, n_rows=100)
    
    print(f"Columns: {df.columns}")
    
    # Calculate text length
    lengths = df["text"].str.len_chars().to_list()
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    
    print(f"Average text length (chars): {avg_len}")
    print(f"Max text length (chars): {max_len}")
    print(f"Sample text (first 200 chars): {df['text'][0][:200]}")

if __name__ == "__main__":
    check_size()
