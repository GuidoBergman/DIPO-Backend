from fastapi import FastAPI, Header
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from os import environ

from schemas import ClassificationInput, ClassificationOutput
from training_evaluation.classificator import Classificator
from middleware import LimitUploadSize, LogRequests

model_name = environ.get('MODEL_NAME')
model_file_name = environ.get('MODEL_FILE_NAME')
evaluation_threshold = float(environ.get('EVALUATION_THRESHOLD'))
max_upload_size = int(environ.get('MAX_UPLOAD_SIZE'))
batch_size = int(environ.get('BATCH_SIZE'))
logging_file = environ.get('LOGGING_FILE')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

app.add_middleware(LimitUploadSize, max_upload_size=max_upload_size)
app.add_middleware(LogRequests)




classificator = Classificator(model_name, model_file_name, evaluation_threshold, batch_size, logging_file)



@app.post("/classify", response_model=ClassificationOutput)
def classify(input: ClassificationInput, content_length: int = Header(..., alias="Content-Length")):
    return classificator.classify(input.text)
    

if __name__ == "__main__":
    uvicorn.run("fastapi_app.app:app", host="0.0.0.0", port=8000, reload=True)