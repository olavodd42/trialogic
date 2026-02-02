import pandas as pd
import os
import csv

SEED = 42

# 1. Load Filtered Dataset
csv_path = os.path.join(os.getcwd(), "data/gold_standard_dataset.csv")
df = pd.read_csv(
    csv_path,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_MINIMAL,
    on_bad_lines='skip'  # pandas >=1.3.0
)

# 2. Define Keywords for bins (Simple Heuristics for Selection)
sepsis_keywords = ["sepsis", "septic", "infection", "fever"]
cardio_keywords = ["chest pain", "acs", "myocardial", "stemi", "troponin"]
trauma_keywords = ["trauma", "fall", "fracture", "mva", "collision", "head injury"]

def filter_cohort(df, keywords, n=50):
    pattern = "|".join(keywords)
    filtered = df[df["text"].str.lower().str.contains(pattern, na=False)]
    sampled = filtered.sample(n=n, random_state=SEED)
    sampled["cohort_type"] = keywords[0]
    return sampled

# 3. Generate the random samples
df_sepsis = filter_cohort(df, sepsis_keywords, n=30)
df_cardio = filter_cohort(df, cardio_keywords, n=30)
df_trauma = filter_cohort(df, trauma_keywords, n=40)

# 4. Concatenate and save
gold_dataset = pd.concat([df_sepsis, df_cardio, df_trauma], ignore_index=True)
gold_dataset = gold_dataset.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print(f"Validation Dataset created with {len(gold_dataset)} notes.")
print(gold_dataset["cohort_type"].value_counts())

gold_dataset.to_csv(os.path.join(os.getcwd(), "data/validation_notes.csv"), index=False)