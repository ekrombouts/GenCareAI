from pathlib import Path

import pandas as pd
from datasets import Dataset

# Read the list of LLM models and their associated ward names
datapath = Path(__file__).resolve().parents[1] / "data"

df_models = pd.read_csv(datapath / "llm_models.csv")

data = {
    "profiles": [],
    "scenarios": [],
    "records": [],
}  # Dictionary to store combined data

# Iterate through each model and its associated ward names
for _, row in df_models.iterrows():
    model = row["llm_model"]  # Get the model name
    for ward_type, folder in [
        ("som_ward_name", "SchilPad"),
        ("pg_ward_name", "PelStraat"),
    ]:
        ward = row[ward_type]  # Get the ward name
        prefix = ward[0]  # Use the first character of the ward name as a prefix

        try:
            # Read the profiles, scenarios, and records CSV files for the current model and ward
            df_p = pd.read_csv(datapath / f"{folder}/profiles_{model}.csv")
            df_s = pd.read_csv(datapath / f"{folder}/scenarios_{model}.csv")
            df_r = pd.read_csv(datapath / f"{folder}/records_{model}.csv")
        except FileNotFoundError:
            # Skip processing if any of the files are missing
            continue

        # Add ward and model information to the profiles DataFrame
        df_p["ward"] = ward
        df_p["model"] = model

        # Update client_id, scenario_id, and note_id with unique prefixes
        df_p["client_id"] = df_p["client_id"].apply(lambda x: f"c{prefix}_{int(x):02d}")
        df_s["client_id"] = df_s["client_id"].apply(lambda x: f"c{prefix}_{int(x):02d}")
        df_r["client_id"] = df_r["client_id"].apply(lambda x: f"c{prefix}_{int(x):02d}")
        df_s["scenario_id"] = df_s["scenario_id"].apply(
            lambda x: f"s{prefix}_{int(x):03d}"
        )
        df_r["scenario_id"] = df_r["scenario_id"].apply(
            lambda x: f"s{prefix}_{int(x):03d}"
        )
        df_r["note_id"] = df_r["note_id"].apply(lambda x: f"n{prefix}_{int(x):04d}")

        # Append the processed DataFrames to the corresponding lists in the data dictionary
        data["profiles"].append(df_p)
        data["scenarios"].append(df_s)
        data["records"].append(df_r)


for name in ["profiles", "scenarios", "records"]:
    df_list = data[name]
    df = pd.concat(df_list)
    df.to_csv(datapath / f"MemoryLane/{name}.csv", index=False)
    # Uncomment the following lines to push the datasets to Hugging Face Hub
    # ds = Dataset.from_pandas(df)
    # ds.push_to_hub(f"ekrombouts/memory_lane_{name}", private=True)
