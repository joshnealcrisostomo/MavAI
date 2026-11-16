# Deploying to the CCS Cloud Linux container

The app runs on the DLSU CCS Cloud LXC so it's a live, accessible demo. These containers are
usually unprivileged, so Docker may not run inside them - the deploy script auto-detects and
falls back to running the app directly with Python.

Our mapped ports (per group): external 2129 -> 22 (SSH), 2149 -> 7860 (frontend/Streamlit),
2169 -> 8000 (backend/FastAPI). The deploy binds the frontend to 7860 and the backend to 8000 so
both are reachable directly on those external ports - no SSH tunnel needed.

## 1. Copy the project onto the server (from your laptop)

A ready-to-ship tarball is created next to the project as `stock-agent-deploy.tar.gz`. Push it up:

```bash
scp -P 2129 stock-agent-deploy.tar.gz patricklim@103.231.240.130:~
```

## 2. On the server: extract and deploy

```bash
ssh -p 2129 patricklim@103.231.240.130
tar xzf stock-agent-deploy.tar.gz
cd stock-agent
bash deploy/deploy_server.sh
```

The script installs what it needs, warms up the reference store, and starts both servers. It
prints the public URLs when it's done.

## 3. Open the app (direct - no tunnel)

- UI:  **http://103.231.240.130:2149**
- API: **http://103.231.240.130:2169/docs**

Backup access if the mapped ports misbehave, via SSH tunnel from your laptop:

```bash
ssh -L 7860:localhost:7860 -L 8000:localhost:8000 -p 2129 patricklim@103.231.240.130
#   then open http://localhost:7860   (API docs: http://localhost:8000/docs)
```

## Notes
- `.env` (with your Groq key) is included in the tarball so it runs immediately. It's your own
  server, but treat the key as a secret.
- The container needs outbound internet (Groq, yfinance, phisix). CCS Cloud containers normally
  have it; if calls fail, that's the first thing to check.
- Redeploy after changes: re-scp the tarball, extract, and re-run the script (it stops the old
  processes first).
