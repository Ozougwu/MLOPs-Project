# Online Shoppers serving image — lean, non-root FastAPI + uvicorn.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_URI=/app/model/champion_model \
    FEATURE_COLUMNS=/app/model/feature_columns.json

WORKDIR /app

# Non-root user for the running process.
RUN adduser --disabled-password --gecos '' mluser

# Install deps first (better layer caching) via uv.
RUN pip install --no-cache-dir uv
COPY serving/requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt --system

# App code + the model artifacts the API loads at startup.
#   - serving/app                  -> /app/app
#   - portable champion model dir  -> /app/model/champion_model
#   - feature schema               -> /app/model/feature_columns.json
COPY serving/app ./app
COPY data/06_models/champion_model ./model/champion_model
COPY data/06_models/feature_columns.json ./model/feature_columns.json

RUN chown -R mluser:mluser /app
USER mluser

EXPOSE 8000
# Container-level liveness probe hitting the same endpoint k8s would use.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
