# Combining data from different models into a single dataset
import pandas as pd

df_models = pd.read_csv("../data/llm_models.csv")

profiles, scenarios, records = [], [], []

for _, row in df_models.iterrows():
    # Get model and ward name
    model, ward = row["llm_model"], row["som_ward_name"]
    # # Uncomment the next line to use the psychogeriatric ward
    # model, ward = row["llm_model"], row["pg_ward_name"]
    prefix = ward[0]

    df_p = pd.read_csv(f"../data/Schilpad/profiles_{model}.csv")
    df_s = pd.read_csv(f"../data/Schilpad/scenarios_{model}.csv")
    df_r = pd.read_csv(f"../data/Schilpad/records_{model}.csv")

    # Add ward name to the profile
    df_p["ward"] = ward
    df_p["model"] = model

    # Make sure the ids are unique, adding the unique first letter of the ward name
    df_p["client_id"] = df_p["client_id"].apply(lambda x: f"c{prefix}_{int(x):02d}")
    df_s["client_id"] = df_s["client_id"].apply(lambda x: f"c{prefix}_{int(x):02d}")
    df_r["client_id"] = df_r["client_id"].apply(lambda x: f"c{prefix}_{int(x):02d}")
    df_s["scenario_id"] = df_s["scenario_id"].apply(lambda x: f"s{prefix}_{int(x):03d}")
    df_r["scenario_id"] = df_r["scenario_id"].apply(lambda x: f"s{prefix}_{int(x):03d}")
    df_r["note_id"] = df_r["note_id"].apply(lambda x: f"n{prefix}_{int(x):04d}")

    profiles.append(df_p)
    scenarios.append(df_s)
    records.append(df_r)

pd.concat(profiles).to_csv("../data/Schilpad/profiles.csv", index=False)
pd.concat(scenarios).to_csv("../data/Schilpad/scenarios.csv", index=False)
pd.concat(records).to_csv("../data/Schilpad/records.csv", index=False)
