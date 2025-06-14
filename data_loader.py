import pandas as pd
from datasets import load_dataset

__all__ = [
    "df",
    "category_enum",
    "intent_enum",
    "CACHE",
]

# ---------------------------------------------------------------------------
# Dataset loading (executed once at import time)
# ---------------------------------------------------------------------------

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
)

df = pd.DataFrame(dataset)

# Enumerations used elsewhere, kept sorted for deterministic ordering
category_enum = sorted(df["category"].unique())
intent_enum = sorted(df["intent"].unique())

# The initial cache contains the full dataframe.  Other modules can mutate
# this reference to share state.
CACHE = df 