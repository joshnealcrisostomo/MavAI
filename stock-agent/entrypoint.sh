#!/bin/sh
# Runs both servers in one container, fronted by nginx on $PORT.
set -e

export PORT="${PORT:-8080}"
# in the Docker/nginx setup the API is served under /api, so tell FastAPI its prefix
export API_ROOT_PATH="${API_ROOT_PATH:-/api}"

# render the nginx config with the runtime port (only $PORT is substituted)
envsubst '${PORT}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# FastAPI channel (internal)
uvicorn api:app --host 127.0.0.1 --port 8000 &

# Streamlit channel (internal)
streamlit run app_streamlit.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false &

# nginx in the foreground keeps the container alive and routes both
exec nginx -g 'daemon off;'
