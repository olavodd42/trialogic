import polars as pl
import os

SEED = 42

# 1. Load Filtered Dataset
df = pl.read_csv(os.path.join(os.getcwd(), "data/discharge_filtered.csv")) # ou o seu arquivo filtrado

# 2. Define Keywords for bins (Simple Heuristics for Selection)
sepsis_keywords = ["sepsis", "septic", "infection", "fever"]
cardio_keywords = ["chest pain", "acs", "myocardial", "stemi", "troponin"]
trauma_keywords = ["trauma", "fall", "fracture", "mva", "collision", "head injury"]

def filter_cohort(df, keywords, n=50):
    pattern = "|".join(keywords)
    
    return (
        df.filter(pl.col("text").str.to_lowercase().str.contains(pattern))
          .sample(n=n, seed=SEED) 
          .with_columns(pl.lit(keywords[0]).alias("cohort_type")) # Marca a coorte
    )

# 3. Generate the random samples
df_sepsis = filter_cohort(df, sepsis_keywords, n=60)
df_cardio = filter_cohort(df, cardio_keywords, n=60)
df_trauma = filter_cohort(df, trauma_keywords, n=120)

# 4. Concatenate and save
gold_dataset = pl.concat([df_sepsis, df_cardio, df_trauma])
gold_dataset = gold_dataset.sample(fraction=1.0, seed=SEED)

print(f"Validation Dataset created with {len(gold_dataset)} notes.")
print(gold_dataset["cohort_type"].value_counts())

gold_dataset.write_csv(os.path.join(os.getcwd(), "data/gold_standard_dataset.csv"))