from typing import Optional

from pydantic import BaseModel


class Agent(BaseModel):
    id: int
    age: int
    gender: int
    ethnicity: int
    infection_rate: Optional[float]
