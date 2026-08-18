#!/usr/bin/env bash
# Levanta la API en Cloud9 / EC2.
# En Cloud9 el puerto expuesto en la vista previa es el 8080; si se accede por
# IP publica se usa el 8000 y debe abrirse ese puerto en el Security Group.
set -euo pipefail

PORT="${PORT:-8000}"

if [ ! -d ".venv" ]; then
  echo "Creando entorno virtual..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements-dev.txt -q

echo "Iniciando servidor en el puerto ${PORT}..."
uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --reload
