# LLM models used in the project

# For generating synthetic data, we use various LLM models.
# This file stores the details of these models in a CSV file, so we can reference them later in our code.
# The ward names are used to refer to the different models in the final dataset.

# Configuration settings for the models are set in: "/src/config/llm_config.py"

from pathlib import Path

import pandas as pd

df_metadata = pd.DataFrame(
    {
        "llm_provider": [
            "azureopenai",
            "azureopenai",
            "anthropic",
            "ollama",
        ],
        "llm_model": [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-5-sonnet-20240620",
            "phi4",
        ],
        "som_ward_name": [
            "appel",
            "banaan",
            "citroen",
            "druif",
        ],
        "pg_ward_name": [
            "kiwi",
            "limoen",
            "mango",
            "nectarine",
        ],
    }
)

output_path = Path(__file__).resolve().parents[1] / "data" / "llm_models.csv"

df_metadata.to_csv(output_path, index=False)
print(f"Models saved to {output_path}")
