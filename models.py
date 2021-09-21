from pydantic import BaseModel

class Query(BaseModel):
    question: str
    class Config:
        schema_extra = {
            "example": {
                "question": "What were PetroTrans interests?",
            }
        }

class ProcessedExhibit(BaseModel):
    exhibit_array: list = []