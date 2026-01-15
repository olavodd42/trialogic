import polars as pl
import os

SEED = 42 # Seed garante reprodutibilidade (CRUCIAL para ciência)

# 1. Carregue o Dataset Filtrado (que você já tem)
# Supondo que você salvou como 'mimic_iv_ed_filtered.csv'
df = pl.read_csv(os.path.join(os.getcwd(), "data/discharge.csv")) # ou o seu arquivo filtrado

# 2. Defina Palavras-chave para as Coortes (Heurística simples para seleção)
# Obs: No mundo ideal usaríamos códigos CID-10 (ICD-10), mas busca textual serve para selecionar notas.
sepsis_keywords = ["sepsis", "septic", "infection", "fever"]
cardio_keywords = ["chest pain", "acs", "myocardial", "stemi", "troponin"]
trauma_keywords = ["trauma", "fall", "fracture", "mva", "collision", "head injury"]

def filter_cohort(df, keywords, n=50):
    # Cria uma expressão regex: "sepsis|septic|infection"
    pattern = "|".join(keywords)
    
    # Filtra e faz amostragem aleatória (Shuffle)
    return (
        df.filter(pl.col("text").str.to_lowercase().str.contains(pattern))
          .sample(n=n, seed=SEED) 
          .with_columns(pl.lit(keywords[0]).alias("cohort_type")) # Marca a coorte
    )

# 3. Gere as Amostras
# Vamos pegar 50 de cada para totalizar 150 casos de teste profundos
df_sepsis = filter_cohort(df, sepsis_keywords, n=15)
df_cardio = filter_cohort(df, cardio_keywords, n=15)
df_trauma = filter_cohort(df, trauma_keywords, n=20)

# 4. Concatene e Salve
gold_dataset = pl.concat([df_sepsis, df_cardio, df_trauma])

# Embaralha final para não ficarem ordenados por tipo
gold_dataset = gold_dataset.sample(fraction=1.0, seed=SEED)

print(f"Dataset de Validação Criado com {len(gold_dataset)} notas.")
print(gold_dataset["cohort_type"].value_counts())

gold_dataset.write_csv(os.path.join(os.getcwd(), "data/gold_standard_dataset.csv"))