FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY data /workspace/data
COPY src /workspace/src

ENV PYTHONPATH=/workspace

CMD ["python", "-c", "print('Thermal interface identification environment ready.')"]
