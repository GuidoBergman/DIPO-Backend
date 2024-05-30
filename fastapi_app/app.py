from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from os import environ

from schemas import ClassificationInput, ClassificationOutput
from training_evaluation.classificator import Classificator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


model_name = environ.get('MODEL_NAME')
model_file_name = environ.get('MODEL_FILE_NAME')
evaluation_threshold = float(environ.get('EVALUATION_THRESHOLD'))
classificator = Classificator(model_name, model_file_name, evaluation_threshold)



@app.post("/classify", response_model=ClassificationOutput)
def classify(input: ClassificationInput):
    return classificator.classify(input.text)
    

if __name__ == "__main__":
    uvicorn.run("fastapi_app.app:app", host="0.0.0.0", port=8000, reload=True)