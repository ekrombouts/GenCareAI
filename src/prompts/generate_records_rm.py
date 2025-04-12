from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class NursesNote(BaseModel):
    date: datetime = Field(description="Datum en tijd van de rapportage")
    note: str = Field(description="Inhoud zorgrapportage")


class ClientRecord(BaseModel):
    record: List[NursesNote]
