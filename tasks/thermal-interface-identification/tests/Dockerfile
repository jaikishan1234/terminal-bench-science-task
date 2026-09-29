FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /app/submission /logs/verifier

COPY . /tests

RUN chmod 700 /tests /logs/verifier && chmod 700 /tests/test.sh
