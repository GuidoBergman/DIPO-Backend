from pydantic import BaseModel



class ClassificationInput(BaseModel):
    text: str

class ClassificationOutput(BaseModel):
    attackOnReputation: list[str]
    manipulativeWording: list[str]