"""Create a validation-notes CSV by removing entries already present in ground_truth."""

import os
import csv

import pandas as pd

SEED = 42

# 1. Load Filtered Dataset
csv_path = os.path.join(os.getcwd(), "data/gold_standard_dataset.csv")
df = pd.read_csv(
    csv_path,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_MINIMAL,
    on_bad_lines='skip'  # pandas >=1.3.0
)

# 2. Load Ground Truth and filter out already processed entries
gt_path = os.path.join(os.getcwd(), "data/ground_truth.csv")
df_gt = pd.read_csv(gt_path, encoding="utf-8-sig")

# Create a set of (subject_id, hadm_id) already in ground_truth
existing_keys = set(zip(df_gt["subject_id"], df_gt["hadm_id"]))
print(f"Entries already in ground_truth: {len(existing_keys)}")

# Keep only rows NOT already in ground_truth
df = df[~df.apply(lambda r: (r["subject_id"], r["hadm_id"]) in existing_keys, axis=1)]
print(f"Remaining candidates after filtering: {len(df)}")

# 3. Save all remaining notes (not in ground_truth)
gold_dataset = df.reset_index(drop=True)

print(f"\nValidation Dataset created with {len(gold_dataset)} notes.")
print(gold_dataset["cohort_type"].value_counts())

gold_dataset.to_csv(os.path.join(os.getcwd(), "data/validation_notes.csv"), index=False)