FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /app/submission

COPY data /app/submission/data
COPY src /app/submission/src

ENV PYTHONPATH=/app/submission

CMD ["python", "-c", "print('Thermal interface identification environment ready.')"]
