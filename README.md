# DIPO - Backend


## Local installation & usage

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

docker build -t 'dipo-backend:latest' .
 docker run -it --rm -p 8000:8000 --env-file .env dipo-backend
