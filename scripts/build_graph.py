import os
import pandas as pd
from src.state.agent_state import AgentState
from src.main import create_workflow

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
DATA_PATH = os.path.join(project_root, "data/gold_standard_dataset.csv")

messages = pd.read_csv(DATA_PATH)

app = create_workflow()
