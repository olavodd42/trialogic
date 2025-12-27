import markdown
import pandas as pd
import polars as pl
from datetime import datetime
from typing import List, Tuple, Dict, TypedDict, Annotated, Optional, Literal
from operator import add
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.schemas.scribe_output_schema import ScribeOutputSchema
from src.schemas.input_schema import InputSchema


llm = ChatOpenAI(name="gpt-4o-mini", temperature=0) 
scribe_model = llm.with_structured_output(ScribeOutputSchema)

with open("prompts/system_prompt.md") as f:
    system_prompt = f.read()

# FIRST NODE
def scribe(state: InputSchema):
    """
    1st Node: The Scribe [cite: 35]
    Receives free text and converts it to a validated structured object.
    """

    print("--- [SCRIBE]: Starting structured extraction ---")
    raw_text = state.get("raw_text", "")


    try:
        structured_data = scribe_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=raw_text)
        ])
        print("--- [SCRIBE]: Structured extraction completed ---")
        return structured_data
    
    except Exception as e:
        print(f"Error during structured extraction: {e}")
        return None