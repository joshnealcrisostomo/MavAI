#!/bin/sh
# Deploy the stock-agent on the CCS Cloud Linux container.
# Run this ON the server, from inside the stock-agent/ folder:  bash deploy/deploy_server.sh
#
# It tries Docker first. If Docker can't run in this LXC (common for unprivileged containers),
# it falls back to running the app directly with Python:
#   Streamlit (frontend) on :7860   and   FastAPI (backend) on :8000
# These match the group's mapped external ports: 2149 -> 7860 (FE) and 2169 -> 8000 (BE).
set -e
cd "$(dirname "$0")/.."   # into stock-agent/

echo "== stock-agent deploy =="

if [ ! -f .env ]; then
  echo "!! .env is missing. Create it with your Groq key, e.g.:"
  echo "   printf 'GROQ_API_KEY=gsk_xxx\\nAGENT_MODEL=llama-3.3-70b-versatile\\nFAST_MODEL=llama-3.1-8b-instant\\n' > .env"
  exit 1
fi

# ---- path 1: Docker, if it actually works in this container ----
if command -v docker >/dev/null 2>&1 && docker run --rm hello-world >/dev/null 2>&1; then
  echo "Docker works here - building and starting via docker compose"
  docker compose up -d --build
  echo ""
  echo "Running. UI + API on port 8080 inside the container."
  echo "From your laptop:  ssh -L 8080:localhost:8080 -p <PORT> <user>@<IP>"
  echo "then open http://localhost:8080"
  exit 0
fi

# ---- path 2: plain Python (reliable in an unprivileged LXC) ----
echo "Docker not usable here - running directly with Python."
if ! command -v python3 >/dev/null 2>&1; then
  echo "installing python..."; sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
fi
sudo apt-get install -y python3-venv python3-pip >/dev/null 2>&1 || true

[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip >/dev/null
echo "installing dependencies (first run takes a couple minutes)..."
pip install -r requirements.txt

# load the env vars for this shell
set -a; . ./.env; set +a

# warm up the reference store (downloads the small embedding model once)
python -c "from core import rag; print('reference docs:', rag.doc_count())"

# stop any previous instances, then start fresh in the background
pkill -f "uvicorn api:app" 2>/dev/null || true
pkill -f "streamlit run app_streamlit.py" 2>/dev/null || true
mkdir -p logs

# frontend/backend ports, matching the group's mapped external ports:
#   external 2149 -> internal 7860 (Streamlit / frontend)
#   external 2169 -> internal 8000 (FastAPI / backend)
FRONTEND_PORT="${FRONTEND_PORT:-7860}"
API_PORT="${API_PORT:-8000}"

GROQ_API_KEY="$GROQ_API_KEY" AGENT_MODEL="$AGENT_MODEL" FAST_MODEL="$FAST_MODEL" \
  nohup .venv/bin/uvicorn api:app --host 0.0.0.0 --port "$API_PORT" > logs/api.log 2>&1 &
# CORS/XSRF off so the UI works when reached directly via the mapped external port (not localhost)
GROQ_API_KEY="$GROQ_API_KEY" AGENT_MODEL="$AGENT_MODEL" FAST_MODEL="$FAST_MODEL" \
  nohup .venv/bin/streamlit run app_streamlit.py \
    --server.port "$FRONTEND_PORT" --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false \
    --server.enableCORS false --server.enableXsrfProtection false > logs/ui.log 2>&1 &

sleep 3
echo ""
echo "Running:"
echo "  Streamlit (frontend) -> internal port $FRONTEND_PORT   (logs/ui.log)"
echo "  FastAPI   (backend)  -> internal port $API_PORT   (logs/api.log)"
echo ""
echo "Public access (no SSH tunnel needed - uses the group's mapped ports):"
echo "  UI  : http://103.231.240.130:2149"
echo "  API : http://103.231.240.130:2169/docs"
echo ""
echo "To stop later:  pkill -f 'streamlit run' ; pkill -f 'uvicorn api:app'"
