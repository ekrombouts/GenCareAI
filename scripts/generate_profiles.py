from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel, Field

from llm.llm_factory import LLMFactory

datapath = Path(__file__).resolve().parents[1] / "data"

# Load llm metadata
df_metadata = pd.read_csv(datapath / "llm_metadata.csv")


# Define a Pydantic model for a single client profile
class ClientProfile(BaseModel):
    naam: str = Field(
        description="naam van de client (Meneer/Mevrouw Voornaam Achternaam, gebruik een naam die je normaal niet zou kiezen)"
    )  # Name of the client
    diagnose: str = Field(
        description="hoofddiagnose voor opname in het verlpleeghuis"
    )  # Diagnosis
    somatiek: str = Field(description="lichamelijke klachten")  # Physical complaints
    adl: str = Field(
        description="beschrijf welke ADL hulp de cliënt nodig heeft"
    )  # ADL assistance
    mobiliteit: str = Field(
        description="beschrijf de mobiliteit (bv rolstoelafhankelijk, gebruik rollator, valgevaar)"
    )  # Mobility description
    gedrag: str = Field(
        description="beschrijf voor de zorg relevante aspecten van cognitie en gedrag."
    )  # Cognitive and behavioral aspects


# Define a Pydantic model to hold multiple client profiles
class ClientProfiles(BaseModel):
    clients: List[ClientProfile]  # List of client profiles


# Define the system prompt for the LLM
system_prompt = (
    """Je bent een behulpzame assistent, die synthetische zorgdata genereert"""
)

# Define the user prompt for generating client profiles
# user_prompt = """Schrijf acht profielen van cliënten die zijn opgenomen op een somatische afdeling van het verpleeghuis. Hier wonen mensen met een hoge zorgzwaarte ten gevolge van een somatische aandoening.
# Zorg dat de profielen erg van elkaar verschillen."""

user_prompt = """Schrijf acht profielen van cliënten die zijn opgenomen op een psychogeriatrische afdeling van het verpleeghuis. Hier wonen mensen met een gevorderde dementie met een hoge zorgzwaarte.
Zorg dat de profielen erg van elkaar verschillen."""

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
            messages=messages,  # Input messages
            model=model,  # LLM model to use
            temperature=0.7,  # Sampling temperature
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
        # Save the DataFrame to a CSV file
        df_profiles.to_csv(output_path, index=False)
    except Exception as e:
        # Handle any errors that occur during the LLM interaction
        print(f"Error with model {model}:", e)
