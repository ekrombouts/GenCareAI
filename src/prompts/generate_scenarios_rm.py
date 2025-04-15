from typing import List

from pydantic import BaseModel, Field


class ClientScenario(BaseModel):
    week: int = Field(description="weeknummer")
    events_description: str = Field(
        description="Korte, duidelijke beschrijving van de week"
    )


class ClientScenarios(BaseModel):
    scenario: List[ClientScenario]
