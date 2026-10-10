#!/bin/sh
set -e

# Render mounts secret files under /etc/secrets. Streamlit reads .streamlit/secrets.toml.
mkdir -p /app/.streamlit
if [ -f /etc/secrets/secrets.toml ]; then
  cp /etc/secrets/secrets.toml /app/.streamlit/secrets.toml
fi

exec python -m streamlit run app.py \
  --server.port="${PORT:-8501}" \
  --server.address=0.0.0.0 \
  --server.headless=true