from pydantic import BaseModel



class ClassificationInput(BaseModel):
    text: str

class ClassificationOutput(BaseModel):
    AttackOnReputation: list[str]
    ManipulativeWording: list[str]