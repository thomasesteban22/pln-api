"""
Pruebas contra un despliegue real (Cloud9/EC2 o Lambda Function URL).

A diferencia de test_endpoints.py, que ejercita la aplicacion en memoria, este
modulo hace peticiones HTTP reales. Sirve como verificacion de despliegue: si
estas pruebas pasan contra ambas URLs, se demuestra que las dos versiones de la
API son funcionalmente equivalentes.

Uso:
    export API_URL=http://<ip-ec2>:8000
    pytest tests/test_remoto.py -v

    export API_URL=https://<id>.lambda-url.us-east-1.on.aws
    pytest tests/test_remoto.py -v

Si API_URL no esta definida, las pruebas se omiten en lugar de fallar.
"""

from __future__ import annotations

import os

import pytest
import requests

API_URL = os.environ.get("API_URL", "").rstrip("/")
TIMEOUT = 60  # Un arranque en frio de Lambda con spaCy puede tardar.

pytestmark = pytest.mark.skipif(
    not API_URL, reason="Defina la variable de entorno API_URL para ejecutar."
)

CORPUS = [
    "gato gato gato comer pescado",
    "juan comer bogota",
    "caballo comer rapido",
]
TEXTO = "Juan Pérez viajó a Bogotá para reunirse con Ecopetrol."


def post(ruta: str, payload: dict) -> dict:
    """Envia una peticion POST y devuelve el cuerpo JSON, validando el codigo."""
    respuesta = requests.post(f"{API_URL}{ruta}", json=payload, timeout=TIMEOUT)
    assert respuesta.status_code == 200, (
        f"{ruta} devolvio {respuesta.status_code}: {respuesta.text[:300]}"
    )
    return respuesta.json()


def test_servicio_disponible():
    """El health check debe responder y reportar el modelo cargado."""
    respuesta = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["modelo_cargado"] is True
    print(f"\nEntorno detectado: {cuerpo['entorno']}")


def test_processed_remoto():
    """El preprocesamiento debe lematizar y eliminar ruido."""
    cuerpo = post("/processed", {"text": "<p>Los gatos corrían</p> 😀"})
    assert "<p>" not in cuerpo["limpieza"]["texto_limpio"]
    assert "correr" in cuerpo["transformacion"]["tokens_procesados"]


def test_dependency_remoto():
    """Debe existir una unica raiz sintactica."""
    cuerpo = post("/dependency", {"text": "El gato come pescado."})
    assert sum(1 for t in cuerpo["tokens"] if t["es_raiz"]) == 1


def test_ner_remoto():
    """Deben reconocerse entidades y devolverse el texto etiquetado."""
    cuerpo = post("/ner", {"text": TEXTO})
    assert cuerpo["num_entidades"] > 0
    assert "[" in cuerpo["texto_etiquetado"]


def test_full_remoto():
    """El analisis completo debe traer las tres secciones."""
    cuerpo = post("/full", {"text": TEXTO})
    for seccion in ["preprocesamiento", "dependencias", "entidades", "resumen"]:
        assert seccion in cuerpo


def test_encoding_remoto():
    """TF-IDF debe penalizar el termino presente en todos los documentos."""
    cuerpo = post("/encoding", {"documents": CORPUS, "method": "tfidf"})
    assert cuerpo["idf"]["comer"] == min(cuerpo["idf"].values())


def test_encoding_comparacion_remoto():
    """La comparacion debe devolver las tres codificaciones."""
    cuerpo = post("/encoding", {"documents": CORPUS, "compare_all": True})
    assert set(cuerpo["resultados"].keys()) == {"one_hot", "bow", "tfidf"}


def test_errores_de_validacion_remoto():
    """El texto vacio debe rechazarse tambien en el despliegue real."""
    respuesta = requests.post(f"{API_URL}/processed", json={"text": ""}, timeout=TIMEOUT)
    assert respuesta.status_code == 422


def test_latencia_aceptable():
    """Tras el arranque en frio, la respuesta debe ser razonablemente rapida."""
    post("/ner", {"text": TEXTO})  # calentamiento
    import time

    inicio = time.perf_counter()
    post("/ner", {"text": TEXTO})
    duracion = time.perf_counter() - inicio
    print(f"\nLatencia en caliente: {duracion * 1000:.0f} ms")
    assert duracion < 10
