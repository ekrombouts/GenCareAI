from typing import List

from pydantic import BaseModel, Field


class ClientScenario(BaseModel):
    week: int = Field(description="Weeknummer")
    events_description: str = Field(
        description="Beschrijving van de gebeurtenissen en zorg"
    )


class ClientScenarios(BaseModel):
    scenario: List[ClientScenario]
