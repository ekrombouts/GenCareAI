from typing import List

from pydantic import BaseModel, Field


# Define a Pydantic model for the note structure. This is a simple list of strings
class Note(BaseModel):
    note: List[str] = Field(description="rapportage tekst")
