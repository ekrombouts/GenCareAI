from pathlib import Path

import pandas as pd

df_metadata = pd.DataFrame(
    {
        "llm_provider": [
            "azureopenai",
            "azureopenai",
            "anthropic",
            "llama",
        ],
        "llm_model": [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-5-sonnet-20240620",
            "phi4",
        ],
    }
)

output_path = Path(__file__).resolve().parents[1] / "data" / "llm_metadata.csv"

df_metadata.to_csv(output_path, index=False)
print(f"Metadata saved to {output_path}")
