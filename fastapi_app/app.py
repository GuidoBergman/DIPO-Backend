from fastapi import FastAPI
import uvicorn
from os import environ

from schemas import ClassificationInput, ClassificationOutput
from training_evaluation.classificator import Classificator

app = FastAPI()


model_name = environ.get('MODEL_NAME')
model_file_name = environ.get('MODEL_FILE_NAME')
evaluation_threshold = float(environ.get('EVALUATION_THRESHOLD'))
print(f'Envs:  {model_name} {model_file_name}  {evaluation_threshold}')
classificator = Classificator(model_name, model_file_name, evaluation_threshold)



@app.post("/classify", response_model=ClassificationOutput)
def classify(input: ClassificationInput):
    techniques = classificator.classify(input.text)
    return {
        'techniques': techniques
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)