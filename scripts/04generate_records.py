# Generate client records based on the generated scenarios and profiles.

# For each model and each client profile, the scenario is transformed into a set of client notes that together
# form a complete record of the client's condition over time.
# The client records contain the following information:
# - client_id: Unique identifier for each client
# - scenario_id: Unique identifier for each scenario
# - date: Date of the record
# - note: The generated record text

# The generated records are saved to a CSV file named records_<model>.csv in the data directory.

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from llm.llm_factory import LLMFactory
from prompts.generate_records_rm import ClientRecord

# --- Configuration ---
datapath = Path(__file__).resolve().parents[1] / "data"
prompts_path = Path(__file__).resolve().parents[1] / "src" / "prompts"

df_models = pd.read_csv(datapath / "llm_models.csv")


# Function to format client profile into a string
def format_naam(row: pd.Series) -> str:
    titel = "Mevrouw" if row["geslacht"] == "v" else "Meneer"
    return f"{titel} {row['voornaam']} {row['achternaam']}"


def format_client_profile(row: pd.Series) -> str:
    return (
        f"Naam: {format_naam(row)}\n"
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

# Loop over llm_models
for _, row_models in df_models.iterrows():
    provider = row_models["llm_provider"]
    model = row_models["llm_model"]

    # Load profiles and scenarios for the specific model
    df_profiles = pd.read_csv(datapath / f"profiles_{model}.csv")
    df_scenarios = pd.read_csv(datapath / f"scenarios_{model}.csv")

    fn_records = datapath / f"records_{model}.csv"

    # Initialize list to store records
    records_list = []
    factory = LLMFactory(provider=provider)

    # Loop over client profiles
    for _, row_profiles in tqdm(
        df_profiles.iterrows(),
        total=len(df_profiles),
        desc=f"Generating records for {model}",
    ):
        # Get profile details to pass to the prompt
        client_name = format_naam(row_profiles)
        client_profile = format_client_profile(row_profiles)
        admission_date = pd.to_datetime(row_profiles["start_date"])

        # Get scenarios for the current client_id
        df_profile_scenarios = df_scenarios[
            df_scenarios["client_id"] == row_profiles["client_id"]
        ]

        # Loop over profile scenarios
        for _, row_profile_scenarios in tqdm(
            df_profile_scenarios.iterrows(),
            total=len(df_profile_scenarios),
            desc=f"Generating records for {client_name}",
        ):

            current_week = row_profile_scenarios["week"]

            past_scenario_list = df_profile_scenarios[
                df_profile_scenarios["week"] < row_profile_scenarios["week"]
            ]["events_description"].tolist()

            past_scenario = "\n".join(past_scenario_list)
            week_scenario = row_profile_scenarios["events_description"]
            start_date = (
                admission_date + pd.Timedelta(weeks=(current_week - 1))
            ).date()
            sex = row_profiles["geslacht"]

            # Render the user prompt using the template
            user_prompt = u_template.render(
                client_profile=client_profile,
                weekno=current_week - 1,
                events_description=past_scenario,
                scenario=week_scenario,
                start_date=start_date,
                dhr_mw="mw." if sex == "v" else "dhr.",
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response_model, _ = factory.create_completion(
                response_model=ClientRecord,
                messages=messages,
                model=model,
            )

            for record in response_model.record:
                records_list.append(
                    (
                        row_profiles["client_id"],
                        row_profile_scenarios["scenario_id"],
                        record.date,
                        record.note,
                    )
                )

            # Save the records to a CSV file after each scenario-line. Prevents loss of data in case of an error or interruption.
            df_records = pd.DataFrame(
                records_list, columns=["client_id", "scenario_id", "date", "note"]
            )
            # Add a note ID column
            df_records.insert(0, "note_id", range(1, len(df_records) + 1))
            df_records.to_csv(fn_records, index=False)
