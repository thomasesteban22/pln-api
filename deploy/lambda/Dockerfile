# Imagen de contenedor para el despliegue serverless en AWS Lambda.
#
# El paquete ZIP de Lambda esta limitado a 250 MB descomprimidos, cifra que
# spaCy y su modelo superan. El despliegue como imagen de contenedor eleva ese
# limite a 10 GB, por lo que no es una preferencia sino un requisito tecnico.
#
# Construccion desde la raiz del repositorio:
#   docker build --platform linux/amd64 --provenance=false \
#     -f deploy/lambda/Dockerfile -t pln-lab:latest .

FROM public.ecr.aws/lambda/python:3.12

# Compiladores necesarios para las dependencias nativas de spaCy y scikit-learn.
RUN dnf install -y gcc gcc-c++ && dnf clean all

COPY requirements.txt /tmp/requirements.txt

# --ignore-installed es imprescindible en esta imagen base.
#
# La imagen de Lambda trae parte de sus dependencias en /var/runtime, ruta que
# forma parte del sys.path del runtime pero no del interprete durante la
# construccion. pip las detecta como ya satisfechas y omite instalarlas en el
# site-packages estandar, de modo que quedan registradas pero no son
# importables. El caso concreto que rompe esta imagen es 'click', requerido por
# typer y por tanto por spaCy: sin la bandera, "import spacy" falla con
# ModuleNotFoundError: No module named 'click'.
RUN pip install --no-cache-dir --ignore-installed \
    -r /tmp/requirements.txt mangum==0.19.0

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

# Verificacion en tiempo de construccion. Comprueba de forma explicita cada
# dependencia critica y el handler completo, de modo que cualquier modulo
# ausente rompa la construccion en lugar de producir un error 500 durante la
# evaluacion. Se informa la ruta de click para dejar constancia de que quedo
# instalado en el site-packages y no solo registrado.
RUN python -c "import click; print('click:', click.__file__)" \
 && python -c "import spacy, sklearn, mangum, fastapi, pydantic; print('dependencias verificadas')" \
 && python -c "import spacy; spacy.load('es_core_news_sm'); print('modelo verificado')" \
 && python -c "import sys; sys.path.insert(0, '${LAMBDA_TASK_ROOT}'); import handler; print('handler verificado')"

CMD ["handler.handler"]
