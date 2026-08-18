# Imagen de contenedor para AWS Lambda.
#
# Se usa la imagen base oficial de AWS para Python, que ya incluye el Runtime
# Interface Client. La construccion es multietapa para que las herramientas de
# compilacion de las dependencias nativas (numpy, scikit-learn, thinc) no
# terminen en la imagen final.
#
# El limite de tamano de una imagen de Lambda es 10 GB, muy por encima de los
# 250 MB del paquete ZIP. Esa es precisamente la razon por la que spaCy se
# despliega como contenedor: el modelo mas scikit-learn superan holgadamente el
# limite del ZIP.

# --------------------------------------------------------------------------
# Etapa 1: construccion de dependencias
# --------------------------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.12 AS builder

RUN dnf install -y gcc gcc-c++ && dnf clean all

COPY requirements.txt .

# --target instala en un directorio aislado que luego se copia completo.
RUN pip install --no-cache-dir -r requirements.txt --target /deps

# --------------------------------------------------------------------------
# Etapa 2: imagen final
# --------------------------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.12

# Dependencias ya compiladas, sin el toolchain de construccion.
COPY --from=builder /deps ${LAMBDA_TASK_ROOT}

# Codigo de la aplicacion.
COPY app ${LAMBDA_TASK_ROOT}/app

# Desactiva la telemetria y fija el directorio de cache en /tmp, el unico
# punto de montaje escribible en Lambda.
ENV SPACY_MODEL=es_core_news_sm \
    HF_HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Verifica en tiempo de construccion que el modelo quedo instalado. Si falta,
# la construccion falla aqui en lugar de en la primera invocacion en produccion.
RUN python -c "import spacy; spacy.load('es_core_news_sm'); print('modelo verificado')"

CMD ["app.lambda_handler.handler"]
