from typing import List

from pydantic import BaseModel, Field


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
