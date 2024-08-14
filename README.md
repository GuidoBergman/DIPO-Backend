# DIPO - Backend


## Local installation & usage

To run you need a .env file with the following variables defined:
- MODEL_NAME: it is used to find the tokenizer
- MODEL_FILE_NAME: the path to model used for classification
- EVALUATION_THRESHOLDS: a comma-separated list of thresholds used for classification. First the one for attack on reputation, and after the one for manipulative wording
- MAX_UPLOAD_SIZE: max size content-length of the requests
- BATCH_SIZE
- LOGGING_FILE: path to a file to log the outputs
- SSL_CERTFILE
- SSL_KEYFILE

Exmaple:
```
MODEL_NAME=xlm-roberta-base
MODEL_FILE_NAME=_model/model.pth
EVALUATION_THRESHOLDS=0.6,0.75
MAX_UPLOAD_SIZE=50000000
BATCH_SIZE=4
LOGGING_FILE=log.csv
```

The paths in SSL_CERTFILE and SSL_KEYFILE must start with certificates/


1. Install dependencies
```
curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry self add poetry-dotenv-plugin
poetry run python -m spacy download en_core_web_sm && \
    poetry run python -m spacy download de_core_news_sm && \
    poetry run python -m spacy download fr_core_news_sm && \
    poetry run python -m spacy download es_core_news_sm
```

2. Start the server:
```
poetry run python fastapi_app/app.py
```




## Run with docker

```
sudo systemctl start docker
sudo docker build -t 'dipo-backend:latest' .
sudo docker run -it --rm -p 8000:8000 --env-file .env dipo-backend
```

The API will be available on `https://localhost:8000`