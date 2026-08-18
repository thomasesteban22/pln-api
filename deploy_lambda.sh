#!/usr/bin/env bash
# Construye la imagen, la publica en ECR y actualiza la funcion Lambda.
#
# Requisitos previos: AWS CLI configurado, Docker en ejecucion y un rol de
# ejecucion de Lambda existente.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
REPO="${ECR_REPO:-pln-api}"
FUNCION="${LAMBDA_FUNCTION:-pln-api}"
CUENTA=$(aws sts get-caller-identity --query Account --output text)
URI="${CUENTA}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"

echo "==> Creando el repositorio de ECR si no existe"
aws ecr describe-repositories --repository-names "${REPO}" --region "${REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${REPO}" --region "${REGION}"

echo "==> Autenticando Docker contra ECR"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${CUENTA}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Construyendo la imagen"
# --provenance=false evita el manifiesto multiplataforma que Lambda rechaza.
docker build --platform linux/amd64 --provenance=false -t "${REPO}:latest" .

echo "==> Publicando en ECR"
docker tag "${REPO}:latest" "${URI}:latest"
docker push "${URI}:latest"

echo "==> Actualizando la funcion Lambda"
if aws lambda get-function --function-name "${FUNCION}" --region "${REGION}" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "${FUNCION}" \
    --image-uri "${URI}:latest" \
    --region "${REGION}"
else
  echo "La funcion no existe. Creela en la consola con:"
  echo "  Tipo: imagen de contenedor    Imagen: ${URI}:latest"
  echo "  Memoria: 2048 MB              Tiempo de espera: 60 s"
  echo "Luego habilite una Function URL con tipo de autenticacion NONE."
fi

echo "==> Listo"
