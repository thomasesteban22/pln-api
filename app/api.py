"""
Microservicio de Procesamiento de Lenguaje Natural.

Implementa el contrato definido en la seccion 8 de la guia del Laboratorio I.
La misma aplicacion se despliega sobre EC2/Cloud9 servida por Uvicorn y sobre
AWS Lambda mediante el adaptador Mangum. La logica funcional es comun a ambas
arquitecturas; los archivos especificos de cada despliegue viven en deploy/.

Rutas del contrato:
    POST /api/v1/clean          application/json
    POST /api/v1/pos            application/json
    POST /api/v1/ner            application/json
    POST /api/v1/visualize/dep  text/html
    POST /api/v1/vectorize      application/json

Universidad Sergio Arboleda
Procesamiento de Lenguaje Natural - Laboratorio I - 2026 S02
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.core import cleaning, linguistics, vectorize
from app.core.pipeline import DEFAULT_MODEL, get_nlp, warmup
from app.schemas import PeticionTexto, PeticionTextoUnico, PeticionVectorizar

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("pln-lab")

PREFIJO = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo de spaCy al arrancar para no penalizar la primera peticion."""
    try:
        warmup()
        logger.info("Modelo '%s' cargado.", DEFAULT_MODEL)
    except OSError as exc:
        logger.error("No se pudo cargar el modelo: %s", exc)
    yield


app = FastAPI(
    title="Microservicio NLP - Laboratorio I",
    description=(
        "Limpieza, analisis POS, reconocimiento de entidades, visualizacion de "
        "dependencias y vectorizacion de texto en espanol."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(ValueError)
async def manejar_value_error(request: Request, exc: ValueError):
    """
    Traduce los errores de validacion de la capa de negocio a respuestas 400.

    Garantiza que ninguna entrada fuera de contrato produzca un error 5xx, que
    la guia considera fallo de la condicion afectada.
    """
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(OSError)
async def manejar_os_error(request: Request, exc: OSError):
    """El modelo no esta disponible: indisponibilidad del servicio."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/", tags=["general"])
def raiz() -> Dict[str, Any]:
    """Describe el servicio y las rutas disponibles."""
    return {
        "service": "Microservicio NLP - Laboratorio I",
        "version": "1.0.0",
        "endpoints": [
            f"POST {PREFIJO}/clean",
            f"POST {PREFIJO}/pos",
            f"POST {PREFIJO}/ner",
            f"POST {PREFIJO}/visualize/dep",
            f"POST {PREFIJO}/vectorize",
        ],
    }


@app.get("/health", tags=["general"])
def health() -> Dict[str, Any]:
    """Comprueba que el servicio responde y que el modelo esta cargado."""
    try:
        get_nlp()
        cargado = True
    except OSError:
        cargado = False

    return {
        "status": "ok" if cargado else "degraded",
        "model": DEFAULT_MODEL,
        "model_loaded": cargado,
        "environment": (
            "lambda" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else "ec2"
        ),
    }


@app.post(f"{PREFIJO}/clean", tags=["nlp"], summary="Limpieza de texto")
def clean(peticion: PeticionTexto) -> Dict[str, Any]:
    """
    Convierte el texto a minusculas, elimina puntuacion y stopwords, y
    normaliza los espacios en blanco.

    El campo cleaned_text es siempre una lista, incluso cuando la entrada es un
    unico string, tal como fija el contrato. La posicion i corresponde al
    documento i de entrada.
    """
    return {"cleaned_text": cleaning.limpiar_lote(peticion.como_lista())}


@app.post(f"{PREFIJO}/pos", tags=["nlp"], summary="Analisis POS y lematizacion")
def pos(peticion: PeticionTexto) -> Dict[str, Any]:
    """
    Devuelve tokens con su categoria gramatical universal y su lema.

    Se analiza el texto original para preservar la informacion sintactica; el
    orden de los tokens y la correspondencia documento a documento se mantienen.
    """
    return {"results": linguistics.analizar_pos(peticion.como_lista())}


@app.post(f"{PREFIJO}/ner", tags=["nlp"], summary="Reconocimiento de entidades")
def ner(peticion: PeticionTexto) -> Dict[str, Any]:
    """
    Detecta entidades nombradas con su texto, tipo y posicion.

    Los indices start y end se refieren al texto original recibido, con start
    inclusivo y end exclusivo.
    """
    return {"results": linguistics.analizar_ner(peticion.como_lista())}


@app.post(
    f"{PREFIJO}/visualize/dep",
    tags=["nlp"],
    summary="Visualizacion de dependencias",
    response_class=HTMLResponse,
)
def visualize_dep(peticion: PeticionTextoUnico) -> HTMLResponse:
    """
    Genera un documento HTML con la representacion SVG del analisis sintactico.

    Procesa un unico documento por solicitud: el uso de lotes en este endpoint
    esta fuera del contrato y se rechaza durante la validacion.
    """
    html = linguistics.visualizar_dependencias(peticion.text)
    return HTMLResponse(content=html, media_type="text/html")


@app.post(f"{PREFIJO}/vectorize", tags=["nlp"], summary="Vectorizacion")
def vectorizar(peticion: PeticionVectorizar) -> Dict[str, Any]:
    """
    Construye el vocabulario comun y las representaciones One-Hot, Bag of Words
    y TF-IDF de la coleccion.

    El vocabulario se devuelve en orden lexicografico ascendente y determina el
    orden de las columnas de las tres representaciones. Las filas conservan el
    orden de los documentos recibidos.
    """
    return vectorize.vectorizar(peticion.documents)
