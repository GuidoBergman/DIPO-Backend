FROM python:3.10
WORKDIR /app
COPY . .
COPY pyproject.toml .
COPY _model  .
RUN pip3 install poetry && poetry config virtualenvs.create false && poetry install --no-dev
EXPOSE 8000
CMD python fastapi_app/app.py