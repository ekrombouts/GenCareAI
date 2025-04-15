from typing import List

from pydantic import BaseModel, Field


# Define a Pydantic model for a single client profile
class ClientProfile(BaseModel):
    geslacht: str = Field(
        description="geslacht van de client (m/v)"
    )  # Gender of the client
    voornaam: str = Field(
        description="voornaam van de client (gebruik een naam die je normaal niet zou kiezen)"
    )  # First name of the client
    achternaam: str = Field(
        description="achternaam van de client (gebruik een naam die je normaal niet zou kiezen)"
    )  # Last name of the client
    diagnose: str = Field(
        description="hoofddiagnose voor opname in het verpleeghuis"
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
