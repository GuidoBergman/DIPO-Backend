# DIPO - Backend


## Local installation & usage

To run you need a .env file with the following variables defined:
- MODEL_NAME
- MODEL_FILE_NAME
- EVALUATION_THRESHOLD
- MAX_UPLOAD_SIZE
- BATCH_SIZE
- LOGGING_FILE
- SSL_CERTFILE
- SSL_KEYFILE

The paths in SSL_CERTFILE and SSL_KEYFILE must start with certificates/

1. Install dependencies
```
curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry self add poetry-dotenv-plugin
```

2. Start the server:
```
poetry run python fastapi_app/app.py
```

The API will be available on `http://localhost:8000/`.


## Run with docker

```
sudo systemctl start docker
sudo docker build -t 'dipo-backend:latest' .
sudo docker run -it --rm -p 8000:8000 --env-file .env dipo-backend
```