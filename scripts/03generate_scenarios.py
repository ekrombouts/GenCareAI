# Generate scenarios for each client profile

# For each model in llm_models.csv, and for each client profile in profiles_{model}.csv, a scenario is generated,
# describing the events that occur during their stay in the nursing home.
# The prompt asks the model to generate a scenario for each week of the client's stay,
# encorporating the complications that have been added to the profile.
# The result is a dataframe with the following columns:
# - scenario_id: Unique identifier for each scenario
# - client_id: Unique identifier for each client
# - week: Week number of the scenario
# - date_start_of_week: Start date of the week
# - events_description: Description of the events that occur during the week
#
# The generated scenarios are saved to a CSV file named scenarios_<model>.csv in the data directory.

import os
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from llm.llm_factory import LLMFactory
from prompts.generate_scenarios_rm import ClientScenarios

datapath = Path(__file__).resolve().parents[1] / "data"
prompts_path = Path(__file__).resolve().parents[1] / "src" / "prompts"

# Load metadata for LLMs (providers and models)
df_models = pd.read_csv(datapath / "llm_models.csv")


# Function to format client profile into a string
def format_naam(row):
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
s_template = env.get_template("generate_scenarios_s.jinja")
system_prompt = s_template.render()  # Render the system prompt template
u_template = env.get_template("generate_scenarios_u.jinja")  # User prompt template

# Iterate over each LLM model row
for _, row_models in df_models.iterrows():
    provider = row_models["llm_provider"]  # Extract LLM provider
    model = row_models["llm_model"]  # Extract model name

    # Load client profiles for the specific model
    fn_profiles = datapath / f"profiles_{model}.csv"
    df_profiles = pd.read_csv(fn_profiles)

    fn_scenarios = datapath / f"scenarios_{model}.csv"

    # Check if the scenarios file already exists
    if not os.path.exists(fn_scenarios):
        scenario_list = []  # Initialize list to store scenarios
        factory = LLMFactory(provider=provider)  # Create LLM factory instance

        # Iterate over each client profile
        for _, row_profiles in tqdm(
            df_profiles.iterrows(),
            total=df_profiles.shape[0],
            desc=f"Generating Scenario's for {model}",
        ):
            # Get profile details to pass to the prompt
            profile = format_client_profile(row_profiles)
            num_weeks = row_profiles["duration"]
            complications = row_profiles["complications"]
            start_date = pd.to_datetime(row_profiles["start_date"])
            sex = row_profiles["geslacht"]

            # Render the user prompt using the template
            user_prompt = u_template.render(
                client_profile=profile,
                num_weeks=num_weeks,
                zijn_haar="haar" if sex == "v" else "zijn",
                complications=complications,
                dhr_mw="mw." if sex == "v" else "dhr.",
            )

            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Generate completion using the LLM
            response_model, _ = factory.create_completion(
                response_model=ClientScenarios,
                messages=messages,
                model=model,
            )

            # Process the response and append scenarios to the list
            for scenario in response_model.scenario:
                scenario_list.append(
                    (
                        row_profiles["client_id"],
                        scenario.week,
                        start_date + pd.Timedelta(weeks=scenario.week),
                        scenario.events_description,
                    )
                )

        # Create a DataFrame from the scenario list
        df_scenarios = pd.DataFrame(
            scenario_list,
            columns=[
                "client_id",
                "week",
                "date_start_of_week",
                "events_description",
            ],
        )
        # Add a scenario ID column
        df_scenarios.insert(0, "scenario_id", range(1, len(df_scenarios) + 1))
        # Save the scenarios to a CSV file
        df_scenarios.to_csv(fn_scenarios, index=False)
        print(f"Data saved to {fn_scenarios}.")
    else:
        # If the file exists, load the data
        print("Scenario file found. Loading data...")
        df_scenarios = pd.read_csv(fn_scenarios)
