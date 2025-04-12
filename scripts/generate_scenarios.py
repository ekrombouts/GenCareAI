import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from llm.llm_factory import LLMFactory
from prompts.generate_scenarios_rm import ClientScenarios

datapath = Path(__file__).resolve().parents[1] / "data"
prompts_path = Path("prompts")

# Load metadata for LLMs (providers and models)
df_llm_metadata = pd.read_csv(datapath / "llm_metadata.csv")


# Function to format client profile into a string
def format_client_profile(row: pd.Series) -> str:
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
s_template = env.get_template("generate_scenarios_s.jinja")
system_prompt = s_template.render()  # Render the system prompt template
u_template = env.get_template("generate_scenarios_u.jinja")  # User prompt template

# Iterate over each LLM metadata row
for _, llm_details in df_llm_metadata.iterrows():
    llm_provider = llm_details["llm_provider"]  # Extract LLM provider
    model_name = llm_details["llm_model"]  # Extract model name

    # Load client profiles for the specific model
    fn_profiles = datapath / f"client_profiles_{model_name}.csv"
    df = pd.read_csv(fn_profiles)

    # Generate a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn_scenarios = datapath / f"client_scenarios_{model_name}_{timestamp}.csv"

    # Check if the scenarios file already exists
    if not os.path.exists(fn_scenarios):
        scenario_list = []  # Initialize list to store scenarios
        factory = LLMFactory(provider=llm_provider)  # Create LLM factory instance

        # Iterate over each client profile
        for _, row in tqdm(
            df.iterrows(),
            total=df.shape[0],
            desc=f"Generating Scenario's for {model_name}",
        ):
            # Format client profile and extract details
            profile = format_client_profile(row)
            num_weeks = row["duration"]
            complications = row["complications"]
            start_date = pd.to_datetime(row["start_date"])

            # Render the user prompt using the template
            user_prompt = u_template.render(
                client_profile=profile,
                num_weeks=num_weeks,
                complications=complications,
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
                model=model_name,
            )

            # Process the response and append scenarios to the list
            for scenario in response_model.scenario:
                scenario_list.append(
                    (
                        row["client_id"],
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
