#!/usr/bin/env bash
# Despliegue persistente sobre la instancia EC2 asociada a AWS Academy/Cloud9.
#
# Levanta el microservicio con Uvicorn. Para acceso desde una red externa hay
# que abrir el puerto en el Security Group de la instancia.
set -euo pipefail

cd "$(dirname "$0")/../.."

PORT="${PORT:-8000}"
WORKERS="${WORKERS:-2}"

if [ ! -d ".venv" ]; then
  echo "==> Creando entorno virtual"
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements-dev.txt -q

echo "==> Iniciando servicio en el puerto ${PORT} con ${WORKERS} workers"
# Varios workers permiten atender solicitudes concurrentes de forma efectiva,
# requisito del atributo de calidad "Concurrencia".
exec uvicorn app.api:app --host 0.0.0.0 --port "${PORT}" --workers "${WORKERS}"
