from pydantic import BaseModel



class ClassificationInput(BaseModel):
    text: str

class ClassificationOutput(BaseModel):
    techniques: list[str]