FROM python:3.6-slim
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY src app/src
COPY model app/model
COPY main.py app/main.py

WORKDIR /app

CMD uvicorn main:app --host 0.0.0.0 --port 8000