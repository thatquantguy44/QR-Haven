FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY infrastructure/volatility-api/runtime-constraints.txt ./runtime-constraints.txt
RUN pip install --no-cache-dir -c runtime-constraints.txt ".[research,deployment]"

ENV QR_HAVEN_VOLATILITY_CANDIDATE_DIR=/model
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN useradd --uid 10001 --create-home qrshadow
USER qrshadow
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "qr_haven.ml.volatility.deployment_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
