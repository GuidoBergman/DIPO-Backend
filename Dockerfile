FROM python:3.10-slim
WORKDIR /app
COPY . .
COPY pyproject.toml .
COPY _model  .
RUN chmod -R 664 certificates/ && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev && \
    poetry run python -m spacy download en_core_web_sm && \
    poetry run python -m spacy download de_core_news_sm && \
    poetry run python -m spacy download fr_core_news_sm && \
    poetry run python -m spacy download es_core_news_sm
    
EXPOSE 8000
CMD python fastapi_app/app.py