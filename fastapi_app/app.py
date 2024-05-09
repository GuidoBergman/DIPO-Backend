from fastapi import FastAPI, Header
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from os import environ

from schemas import ClassificationInput, ClassificationOutput
from training_evaluation.classificator import Classificator
from middleware import LimitUploadSize


model_name = environ.get('MODEL_NAME')
model_file_name = environ.get('MODEL_FILE_NAME')
evaluation_thresholds = [float(i) for i in environ.get('EVALUATION_THRESHOLDS').split(',')]
max_upload_size = int(environ.get('MAX_UPLOAD_SIZE'))
batch_size = int(environ.get('BATCH_SIZE'))
logging_file = environ.get('LOGGING_FILE')
ssl_keyfile = environ.get('SSL_KEYFILE')
ssl_certfile = environ.get('SSL_CERTFILE')

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Length"],
)

app.add_middleware(LimitUploadSize, max_upload_size=max_upload_size)



classificator = Classificator(model_name, model_file_name, evaluation_thresholds, batch_size, logging_file)



@app.post("/classify", response_model=ClassificationOutput)
def classify(input: ClassificationInput, content_length: int = Header(..., alias="Content-Length")):
    return classificator.classify(input.text)
    

if __name__ == "__main__":
    uvicorn.run("fastapi_app.app:app", host="0.0.0.0", port=8000, reload=True, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)