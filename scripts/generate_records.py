import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from llm.llm_factory import LLMFactory
from prompts.generate_records_rm import ClientRecord

# --- Configuration ---
datapath = Path(__file__).resolve().parents[1] / "data"
prompts_path = Path("prompts")
df_llm_metadata = pd.read_csv(datapath / "llm_metadata.csv")
df_profiles = pd.read_csv(datapath / "client_profiles_gpt-4o-mini.csv")
df_scenarios = pd.read_csv(datapath / "client_scenarios_gpt-4o-mini.csv")


def format_client_profile(row):
    return (
        f"Naam: {row['naam']}\n"
        f"Diagnose: {row['diagnose']}\n"
        f"Lichamelijke klachten: {row['somatiek']}\n"
        f"ADL: {row['adl']}\n"
        f"Mobiliteit: {row['mobiliteit']}\n"
        f"Gedrag: {row['gedrag']}"
    )


# Load the Jinja2 templates for prompts
env = Environment(loader=FileSystemLoader(prompts_path))
s_template = env.get_template("generate_records_s.jinja")
system_prompt = s_template.render()
u_template = env.get_template("generate_records_u.jinja")

# Loop over llm_metadata
for _, row in df_llm_metadata.iterrows():
    llm_provider = row["llm_provider"]
    model_name = row["llm_model"]
    factory = LLMFactory(provider=llm_provider)
    records_list = []

    # Loop over client profiles
    for _, profile_row in df_profiles.iterrows():
        client_profile = format_client_profile(profile_row)
        admission_date = pd.to_datetime(profile_row["start_date"])
        print(profile_row["naam"])

        # Get scenarios for the current client_id
        df_client_scenarios = df_scenarios[
            df_scenarios["client_id"] == profile_row["client_id"]
        ]

        # Loop over client scenarios
        for _, scenario_row in df_client_scenarios.iterrows():

            current_week = scenario_row["week"]

            events_description_list = df_client_scenarios[
                df_client_scenarios["week"] < scenario_row["week"]
            ]["events_description"].tolist()

            events_upto_now = "\n".join(events_description_list)
            week_scenario = scenario_row["events_description"]
            print(week_scenario)
            start_date = (
                admission_date + pd.Timedelta(weeks=(current_week - 1))
            ).date()
            user_prompt = u_template.render(
                client_profile=client_profile,
                weekno=current_week - 1,
                events_description=events_upto_now,
                scenario=week_scenario,
                start_date=start_date,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response_model, _ = factory.create_completion(
                response_model=ClientRecord,
                messages=messages,
                model=model_name,
            )

            for record in response_model.record:
                records_list.append(
                    (
                        profile_row["client_id"],
                        scenario_row["scenario_id"],
                        record.date,
                        record.note,
                    )
                )
    df_out = pd.DataFrame(
        records_list, columns=["client_id", "scenario_id", "date", "note"]
    )
    output_dir = datapath / "generated_records"
    output_dir.mkdir(exist_ok=True)
    df_out.to_csv(output_dir / f"records_{model_name}.csv", index=False)
