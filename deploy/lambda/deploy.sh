#!/usr/bin/env bash
# Despliegue serverless en AWS Lambda mediante imagen de contenedor.
#
# Construye la imagen, la publica en ECR, crea o actualiza la funcion y expone
# una Function URL publica.
#
# Ejecutar desde la raiz del repositorio:  ./deploy/lambda/deploy.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
REPO="${ECR_REPO:-pln-lab}"
FUNCION="${LAMBDA_FUNCTION:-pln-lab}"
MEMORIA="${LAMBDA_MEMORY:-2048}"
TIMEOUT="${LAMBDA_TIMEOUT:-60}"

CUENTA=$(aws sts get-caller-identity --query Account --output text)
URI="${CUENTA}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"
ROL="arn:aws:iam::${CUENTA}:role/LabRole"

echo "==> Repositorio de ECR"
aws ecr describe-repositories --repository-names "${REPO}" --region "${REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${REPO}" --region "${REGION}" >/dev/null

echo "==> Autenticando Docker contra ECR"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${CUENTA}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Construyendo la imagen"
# --provenance=false evita el manifiesto multiplataforma que Lambda rechaza.
# --platform linux/amd64 fuerza la arquitectura de la funcion.
docker build --platform linux/amd64 --provenance=false \
  -f deploy/lambda/Dockerfile -t "${REPO}:latest" .

echo "==> Publicando en ECR"
docker tag "${REPO}:latest" "${URI}:latest"
docker push "${URI}:latest"

echo "==> Creando o actualizando la funcion"
if aws lambda get-function --function-name "${FUNCION}" --region "${REGION}" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "${FUNCION}" --image-uri "${URI}:latest" --region "${REGION}" >/dev/null
else
  aws lambda create-function \
    --function-name "${FUNCION}" \
    --package-type Image \
    --code "ImageUri=${URI}:latest" \
    --role "${ROL}" \
    --memory-size "${MEMORIA}" \
    --timeout "${TIMEOUT}" \
    --architectures x86_64 \
    --region "${REGION}" >/dev/null
fi

aws lambda wait function-updated --function-name "${FUNCION}" --region "${REGION}" 2>/dev/null || true
aws lambda wait function-active --function-name "${FUNCION}" --region "${REGION}" 2>/dev/null || true

echo "==> Configurando la Function URL"
aws lambda get-function-url-config --function-name "${FUNCION}" --region "${REGION}" >/dev/null 2>&1 \
  || aws lambda create-function-url-config \
       --function-name "${FUNCION}" --auth-type NONE --region "${REGION}" >/dev/null

# Desde octubre de 2025 las Function URL requieren AMBOS permisos en la politica
# basada en recursos. Con solo lambda:InvokeFunctionUrl la URL devuelve 403
# aunque el tipo de autenticacion sea NONE.
aws lambda add-permission \
  --function-name "${FUNCION}" --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl --principal "*" \
  --function-url-auth-type NONE --region "${REGION}" >/dev/null 2>&1 || true

aws lambda add-permission \
  --function-name "${FUNCION}" --statement-id FunctionURLAllowPublicInvoke \
  --action lambda:InvokeFunction --principal "*" --region "${REGION}" >/dev/null 2>&1 || true

URL=$(aws lambda get-function-url-config --function-name "${FUNCION}" \
        --region "${REGION}" --query FunctionUrl --output text)

echo
echo "==> Despliegue completado"
echo "    Function URL: ${URL}"
echo
echo "    Verificacion:"
echo "      curl -s ${URL%/}/health"
