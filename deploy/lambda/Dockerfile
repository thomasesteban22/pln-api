# Imagen de contenedor para el despliegue serverless en AWS Lambda.
#
# El paquete ZIP de Lambda esta limitado a 250 MB descomprimidos, cifra que
# spaCy y su modelo superan. El despliegue como imagen de contenedor eleva ese
# limite a 10 GB, por lo que no es una preferencia sino un requisito tecnico.
#
# Construccion desde la raiz del repositorio:
#   docker build --platform linux/amd64 --provenance=false \
#     -f deploy/lambda/Dockerfile -t pln-lab:latest .

FROM public.ecr.aws/lambda/python:3.12 AS builder

RUN dnf install -y gcc gcc-c++ && dnf clean all

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt mangum==0.19.0 --target /deps

FROM public.ecr.aws/lambda/python:3.12

COPY --from=builder /deps ${LAMBDA_TASK_ROOT}

# Logica funcional comun a ambas arquitecturas.
COPY app ${LAMBDA_TASK_ROOT}/app

# Punto de entrada especifico de esta arquitectura.
COPY deploy/lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py

# En Lambda solo /tmp es escribible; se redirigen alli los directorios de cache.
ENV SPACY_MODEL=es_core_news_sm \
    HF_HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Verificacion en tiempo de construccion: si el modelo no quedo instalado, la
# construccion falla aqui y no en la primera invocacion en produccion.
RUN python -c "import spacy; spacy.load('es_core_news_sm'); print('modelo verificado')"

CMD ["handler.handler"]
