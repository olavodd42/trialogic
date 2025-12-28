import polars as pl

from src.agents.scribe import scribe
from src.schemas.input_schema import InputSchema


df = pl.read_csv("data/discharge.csv")
outputs = []
for row in df.iter_rows(named=True):
    input_data = InputSchema(
        subject_id=row["subject_id"],
        hadm_id=row["hadm_id"],
        raw_text=row["text"]
    )
    
    structured_output = scribe(input_data)
    outputs.append({
        "subject_id": input_data["subject_id"],
        "hadm_id": input_data["hadm_id"],
        "structured_output": structured_output
    })