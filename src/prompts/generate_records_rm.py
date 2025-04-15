from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class NursesNote(BaseModel):
    date: datetime = Field(
        description="Datum en tijdstip waarop de rapportage is geschreven"
    )
    note: str = Field(description="Inhoud van de rapportage")


class ClientRecord(BaseModel):
    record: List[NursesNote]
