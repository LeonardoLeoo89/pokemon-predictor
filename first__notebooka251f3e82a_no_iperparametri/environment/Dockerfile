FROM ghcr.io/astral-sh/uv@sha256:4f52717de41541452f5318571b05da783d2ddf346a94b4d5c6512a4b51e986bd
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PYTHONPATH=/app PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENV UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=2
COPY pyproject.toml uv.lock ./
RUN uv python install 3.13 && uv sync --python 3.13 --frozen --inexact --no-install-project
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["-m", "src.pipeline"]

RUN uv pip install --python /opt/venv/bin/python scikit-learn==1.7.2 joblib==1.5.2 threadpoolctl==3.6.0
ENV PYTHONHASHSEED=42
