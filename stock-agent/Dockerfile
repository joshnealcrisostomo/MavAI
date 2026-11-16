# UNBUILD?: docker compose down
# BUILD: docker compose build
# RUN: docker compose up
# Website local host: http://localhost:8080

# stage 1: build the dependency venv
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# drop heavy bits streamlit/mlflow pull but we never use (we chart with plotly, not altair)
RUN set -eux; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    rm -rf "$SP"/kubernetes "$SP"/pydeck "$SP"/altair "$SP"/pip "$SP"/pip-* "$SP"/wheel "$SP"/wheel-* ; \
    find "$SP" -type d -name '__pycache__' -prune -exec rm -rf {} + ; \
    find "$SP" -type d \( -name 'tests' -o -name 'test' \) -prune -exec rm -rf {} + ; \
    find "$SP" \( -name '*.pyc' -o -name '*.pyx' \) -delete

# stage 2: slim runtime
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"

# nginx-light (router) + gettext-base (envsubst) + curl (healthcheck)
RUN apt-get update \
 && apt-get install -y --no-install-recommends nginx-light gettext-base curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY core/ ./core/
COPY app_streamlit.py api.py ./
COPY nginx.conf.template /etc/nginx/nginx.conf.template
COPY entrypoint.sh /entrypoint.sh
# strip any CR (Windows CRLF) so the shebang works regardless of how the file was checked out
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# warm up the RAG store at build time: this downloads the ONNX embedding model and seeds the
# Chroma reference store INTO the image, so the first query at demo time is fast and needs no
# network for embeddings.
RUN python -c "from core import rag; print('seeded', rag.doc_count(), 'reference docs')"

ENV PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fs http://localhost:8080/api/health || exit 1

# call sh explicitly so we don't depend on the exec bit surviving a Windows checkout
CMD ["sh", "/entrypoint.sh"]
