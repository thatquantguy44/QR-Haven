FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[research,deployment]"

ENV QR_HAVEN_VOLATILITY_CANDIDATE_DIR=/model
EXPOSE 8000
CMD ["uvicorn", "qr_haven.ml.volatility.deployment_api:app", "--host", "0.0.0.0", "--port", "8000"]
