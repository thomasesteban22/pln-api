"""
API de Procesamiento de Lenguaje Natural.

Expone los servicios de preprocesamiento, analisis sintactico, reconocimiento
de entidades y codificacion de texto sobre una interfaz REST.

La misma aplicacion se despliega en dos entornos:
  - Cloud9 / EC2, servida directamente por Uvicorn.
  - AWS Lambda, mediante el adaptador Mangum sobre una imagen de contenedor.

El codigo de la aplicacion es identico en ambos casos; solo cambia el punto de
entrada (ver app/lambda_handler.py).

Universidad Sergio Arboleda
Programa de Ciencias de la Computacion e Inteligencia Artificial
Procesamiento de Lenguaje Natural (PCIA5011)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import spacy
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core import encoding, linguistics, preprocessing
from app.core.pipeline import DEFAULT_MODEL, get_nlp, warmup
from app.schemas import (
    PeticionCodificacion,
    PeticionPreprocesamiento,
    PeticionTexto,
    RespuestaSalud,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("pln-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga el modelo de spaCy durante el arranque de la aplicacion.

    Si la carga falla, se registra el error pero el proceso continua: el
    endpoint /health quedara disponible para diagnosticar el problema en lugar
    de que el contenedor entre en un ciclo de reinicios.
    """
    inicio = time.perf_counter()
    try:
        warmup()
        logger.info(
            "Modelo '%s' cargado en %.2f s", DEFAULT_MODEL, time.perf_counter() - inicio
        )
    except OSError as exc:
        logger.error("No se pudo cargar el modelo de spaCy: %s", exc)
    yield
    logger.info("Apagando la aplicacion.")


app = FastAPI(
    title="API de Procesamiento de Lenguaje Natural",
    description=(
        "Servicios de preprocesamiento, analisis de dependencias, "
        "reconocimiento de entidades nombradas y codificacion de texto. "
        "Universidad Sergio Arboleda - PCIA5011."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def agregar_tiempo_proceso(request: Request, call_next):
    """Anade la latencia de la peticion como cabecera de respuesta."""
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    duracion_ms = (time.perf_counter() - inicio) * 1000
    respuesta.headers["X-Process-Time-Ms"] = f"{duracion_ms:.2f}"
    return respuesta


@app.exception_handler(ValueError)
async def manejar_value_error(request: Request, exc: ValueError):
    """Convierte los ValueError de la capa de negocio en respuestas 400."""
    return JSONResponse(
        status_code=400, content={"detail": str(exc), "tipo": "ValueError"}
    )


@app.exception_handler(OSError)
async def manejar_os_error(request: Request, exc: OSError):
    """El modelo de spaCy no esta disponible: es un fallo del servicio (503)."""
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "tipo": "ModeloNoDisponible"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["general"], summary="Informacion del servicio")
def raiz() -> Dict[str, Any]:
    """Devuelve el catalogo de endpoints disponibles."""
    return {
        "servicio": "API de Procesamiento de Lenguaje Natural",
        "version": "1.0.0",
        "asignatura": "PCIA5011 - Procesamiento de Lenguaje Natural",
        "institucion": "Universidad Sergio Arboleda",
        "endpoints": {
            "POST /processed": "Limpieza y transformacion de texto con spaCy.",
            "POST /dependency": "Arbol de dependencias sintacticas.",
            "POST /ner": "Reconocimiento de entidades nombradas.",
            "POST /full": "Analisis completo: los tres anteriores combinados.",
            "POST /encoding": "Codificacion one-hot, BoW o TF-IDF.",
            "GET /health": "Estado del servicio y del modelo.",
            "GET /docs": "Documentacion interactiva OpenAPI.",
        },
    }


@app.get("/health", response_model=RespuestaSalud, tags=["general"])
def salud() -> Dict[str, Any]:
    """
    Verifica que el servicio responde y que el modelo esta cargado.

    Se usa como health check del contenedor y como prueba de humo tras cada
    despliegue.
    """
    try:
        get_nlp()
        cargado = True
    except OSError:
        cargado = False

    return {
        "status": "ok" if cargado else "degradado",
        "modelo": DEFAULT_MODEL,
        "modelo_cargado": cargado,
        "version_spacy": spacy.__version__,
        "entorno": "lambda" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else "server",
    }


@app.post("/processed", tags=["nlp"], summary="Limpieza y transformacion")
def procesado(peticion: PeticionPreprocesamiento) -> Dict[str, Any]:
    """
    Aplica limpieza y transformacion sobre el texto recibido.

    La limpieza elimina HTML, URLs, correos y emojis; la transformacion
    tokeniza, descarta stop words y puntuacion, y lematiza. La respuesta
    incluye la traza completa de las operaciones aplicadas y el detalle de los
    tokens descartados con su motivo, de modo que el resultado sea auditable.
    """
    resultado = preprocessing.process(
        peticion.text, model=peticion.model, **peticion.options.model_dump()
    )
    return {"endpoint": "/processed", **resultado}


@app.post("/dependency", tags=["nlp"], summary="Analisis de dependencias")
def dependencias(peticion: PeticionTexto) -> Dict[str, Any]:
    """
    Devuelve el arbol de dependencias sintacticas.

    Opera sobre el texto crudo de forma deliberada: la puntuacion es la senal
    principal que utiliza el parser para delimitar oraciones y clausulas.
    """
    resultado = linguistics.analyze_dependencies(peticion.text, model=peticion.model)
    return {"endpoint": "/dependency", **resultado}


@app.post("/ner", tags=["nlp"], summary="Reconocimiento de entidades nombradas")
def entidades(peticion: PeticionTexto) -> Dict[str, Any]:
    """
    Reconoce entidades nombradas y devuelve el texto con las etiquetas.

    Se entrega tanto la lista estructurada con offsets como el texto anotado en
    linea y el esquema BIO por token.
    """
    resultado = linguistics.analyze_entities(peticion.text, model=peticion.model)
    return {"endpoint": "/ner", **resultado}


@app.post("/full", tags=["nlp"], summary="Analisis completo")
def completo(peticion: PeticionPreprocesamiento) -> Dict[str, Any]:
    """
    Ejecuta preprocesamiento, analisis de dependencias y NER sobre el mismo texto.

    Los analisis linguisticos se aplican al texto original y no al preprocesado:
    lematizar y eliminar stop words antes del parser o del reconocedor de
    entidades degrada notablemente ambos resultados.
    """
    inicio = time.perf_counter()

    preproceso = preprocessing.process(
        peticion.text, model=peticion.model, **peticion.options.model_dump()
    )
    sintaxis = linguistics.analyze_dependencies(peticion.text, model=peticion.model)
    ner = linguistics.analyze_entities(peticion.text, model=peticion.model)

    return {
        "endpoint": "/full",
        "texto_original": peticion.text,
        "preprocesamiento": preproceso,
        "dependencias": sintaxis,
        "entidades": ner,
        "resumen": {
            "num_tokens": sintaxis["num_tokens"],
            "num_oraciones": sintaxis["num_oraciones"],
            "num_entidades": ner["num_entidades"],
            "tokens_tras_preprocesamiento": preproceso["transformacion"][
                "num_tokens_procesados"
            ],
            "tamano_vocabulario": preproceso["transformacion"]["tamano_vocabulario"],
            "tiempo_ms": round((time.perf_counter() - inicio) * 1000, 2),
        },
    }


@app.post("/encoding", tags=["nlp"], summary="Codificacion de texto")
def codificacion(peticion: PeticionCodificacion) -> Dict[str, Any]:
    """
    Codifica una coleccion de documentos como vectores numericos.

    Soporta one-hot (presencia/ausencia), Bag-of-Words (frecuencia absoluta) y
    TF-IDF, con n-gramas configurables. Si compare_all es true se devuelven las
    tres representaciones sobre el mismo corpus, lo que permite contrastarlas
    termino a termino.
    """
    if peticion.ngram_max < peticion.ngram_min:
        raise HTTPException(
            status_code=400,
            detail="ngram_max debe ser mayor o igual que ngram_min.",
        )

    documentos = peticion.documents

    if peticion.preprocess:
        documentos = [
            preprocessing.process(doc)["transformacion"]["texto_procesado"] or doc
            for doc in documentos
        ]

    rango = (peticion.ngram_min, peticion.ngram_max)

    if peticion.compare_all:
        resultado = encoding.comparar_metodos(documentos, ngram_range=rango)
        return {
            "endpoint": "/encoding",
            "preprocesado": peticion.preprocess,
            **resultado,
        }

    resultado = encoding.encode(
        documentos,
        metodo=peticion.method,
        ngram_range=rango,
        max_features=peticion.max_features,
    )
    return {
        "endpoint": "/encoding",
        "preprocesado": peticion.preprocess,
        "corpus": documentos,
        **resultado,
    }
