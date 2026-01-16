import pandas as pd
import os

files = [
    "data/master_dataset.csv",
    "data/discharge.csv",
    "data/discharge_filtered.csv"
]

target_hadm_id = 27598532
target_subject_id = 14037648

print(f"Checking for HADM_ID: {target_hadm_id} and SUBJECT_ID: {target_subject_id}")

for f in files:
    if not os.path.exists(f):
        print(f"Skipping {f} (not found)")
        continue
        
    print(f"Scanning {f}...")
    try:
        # Read in chunks to handle large files
        found = False
        for chunk in pd.read_csv(f, chunksize=10000):
            # Check for column existence first
            if 'hadm_id' not in chunk.columns:
                print(f"  Warning: 'hadm_id' not in {f}")
                break
                
            # Check for ID match
            match = chunk[chunk['hadm_id'] == target_hadm_id]
            if not match.empty:
                print(f"  FOUND in {f}!")
                print(match[['subject_id', 'hadm_id']].to_string())
                found = True
                break
        
        if not found:
            print(f"  Not found in {f}")
            
    except Exception as e:
        print(f"  Error reading {f}: {e}")
