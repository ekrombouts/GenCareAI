import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from llm.llm_factory import LLMFactory
from prompts.generate_profiles_rm import ClientProfiles

# llm settings are loaded from the config module (default model, temperature, etc.)

datapath = Path(__file__).resolve().parents[1] / "data"
prompts_path = Path("prompts")

# Load llm metadata
df_metadata = pd.read_csv(datapath / "llm_metadata.csv")

complications_library = [
    "gewichtsverlies",
    "algehele achteruitgang",
    "decubitus",
    "urineweginfectie",
    "pneumonie",
    "delier",
    "verergering van onderliggende lichamelijke klachten",
    "verbetering van de klachten",
    "overlijden",
    "valpartij",
]


def pick_start_date(from_date="2024-01-01", to_date="2025-01-01"):
    from_date = datetime.strptime(from_date, "%Y-%m-%d")
    to_date = datetime.strptime(to_date, "%Y-%m-%d")
    return from_date + (to_date - from_date) * random.random()


def determine_duration(mean=10, std_dev=4):
    return int(np.round(np.random.normal(mean, std_dev)))


def sample_complications(complications_library, min_n=1, max_n=3):
    num = random.randint(min_n, max_n)
    return ", ".join(random.sample(complications_library, num))


# Load the prompts
env = Environment(loader=FileSystemLoader(prompts_path))
s_template = env.get_template("generate_profiles_s.jinja")
system_prompt = s_template.render()
u_template = env.get_template("generate_profiles_u.jinja")
user_prompt = u_template.render(
    # profile_type="somatische afdeling",
    profile_type="psychogeriatrische afdeling",
    # description="mensen met een hoge zorgzwaarte ten gevolge van een somatische aandoening",
    description="mensen met een gevorderde dementie met een hoge zorgzwaarte",
)

# Combine the system and user prompts into a list of messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# Iterate over each row in the metadata DataFrame
for _, row in df_metadata.iterrows():
    provider = row["llm_provider"]  # Extract the LLM provider
    model = row["llm_model"]  # Extract the LLM model

    # Create an instance of the LLMFactory for the given provider
    factory = LLMFactory(provider=provider)

    try:
        # Generate client profiles using the LLM
        response_model, raw_response = factory.create_completion(
            response_model=ClientProfiles,  # Expected response model
            model=model,  # LLM model to use, overrides the default model from llm_config
            messages=messages,  # Input messages
        )
        print(f"Parsed Response for {model}:", response_model)
        # Uncomment the line below to print raw LLM responses for debugging
        # print(f"Raw Response for {model}:", raw_response)

        # Convert the generated client profiles to a pandas DataFrame
        df_profiles = pd.DataFrame(
            [profile.model_dump() for profile in response_model.clients]
        )
        # Define the output file path for saving the profiles
        # Add current date and time to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = datapath / f"client_profiles_{model}_{timestamp}.csv"

        # Add a unique client ID to each profile
        df_profiles.insert(0, "client_id", range(1, len(df_profiles) + 1))

        # Add a start date to each profile
        df_profiles["start_date"] = [
            pick_start_date(from_date="2024-01-01", to_date="2025-01-01")
            for _ in range(len(df_profiles))
        ]
        # Add a duration to each profile
        df_profiles["duration"] = [
            determine_duration(mean=10, std_dev=4) for _ in range(len(df_profiles))
        ]
        # Add complications to each profile
        df_profiles["complications"] = [
            sample_complications(complications_library, 1, 3)
            for _ in range(len(df_profiles))
        ]

        # Save the DataFrame to a CSV file
        df_profiles.to_csv(output_path, index=False)
    except Exception as e:
        # Handle any errors that occur during the LLM interaction
        print(f"Error with model {model}:", e)
