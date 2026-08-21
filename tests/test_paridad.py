"""
Verificacion de paridad entre los dos despliegues.

El atributo de calidad "Paridad" exige que una misma solicitud produzca
resultados funcionalmente equivalentes en EC2 y en Lambda. Esta suite envia las
mismas peticiones a ambas URLs y compara las respuestas.

Uso:
    export EC2_URL=http://<ip-publica>:8000
    export LAMBDA_URL=https://<id>.lambda-url.us-east-1.on.aws
    pytest tests/test_paridad.py -v

Si alguna de las dos variables no esta definida, las pruebas se omiten.
"""

from __future__ import annotations

import os

import pytest

requests = pytest.importorskip("requests")

EC2_URL = os.environ.get("EC2_URL", "").rstrip("/")
LAMBDA_URL = os.environ.get("LAMBDA_URL", "").rstrip("/")
PREFIJO = "/api/v1"
TIMEOUT = 90

pytestmark = pytest.mark.skipif(
    not (EC2_URL and LAMBDA_URL),
    reason="Defina EC2_URL y LAMBDA_URL para ejecutar la verificacion de paridad.",
)

CORPUS = [
    "El gato come pescado y el gato duerme.",
    "Juan Pérez come en Bogotá con Ecopetrol.",
    "El caballo corría rápido, pero se detuvo.",
]

CASOS = [
    (f"{PREFIJO}/clean", {"text": "El gato come pescado."}),
    (f"{PREFIJO}/clean", {"text": CORPUS}),
    (f"{PREFIJO}/clean", {"text": "casa,perro. gato;pez ñandú 25%"}),
    (f"{PREFIJO}/pos", {"text": "Los gatos corrían rápidamente por los tejados"}),
    (f"{PREFIJO}/pos", {"text": CORPUS}),
    (f"{PREFIJO}/ner", {"text": "Juan Pérez viajó a Bogotá con Ecopetrol."}),
    (f"{PREFIJO}/ner", {"text": CORPUS}),
    (f"{PREFIJO}/vectorize", {"documents": CORPUS}),
    (f"{PREFIJO}/vectorize", {"documents": ["primer documento", "segundo documento"]}),
]

INVALIDOS = [
    (f"{PREFIJO}/clean", {"text": ""}),
    (f"{PREFIJO}/clean", {"text": []}),
    (f"{PREFIJO}/pos", {"text": None}),
    (f"{PREFIJO}/visualize/dep", {"text": ["uno", "dos"]}),
    (f"{PREFIJO}/vectorize", {"documents": ["uno"]}),
]


def pedir(base: str, ruta: str, payload):
    """Envia una peticion POST a una de las dos URLs."""
    return requests.post(f"{base}{ruta}", json=payload, timeout=TIMEOUT)


@pytest.fixture(scope="session", autouse=True)
def calentar():
    """
    Invoca ambos despliegues antes de medir.

    La guia excluye explicitamente la primera invocacion en frio de Lambda de la
    medicion ordinaria de rendimiento, de modo que el calentamiento previo es
    parte del procedimiento correcto de verificacion.
    """
    for base in (EC2_URL, LAMBDA_URL):
        try:
            pedir(base, f"{PREFIJO}/clean", {"text": "calentamiento"})
        except requests.RequestException:
            pass


def test_ambos_despliegues_responden():
    """Las dos URLs deben estar accesibles."""
    for nombre, base in (("EC2", EC2_URL), ("Lambda", LAMBDA_URL)):
        r = requests.get(f"{base}/health", timeout=TIMEOUT)
        assert r.status_code == 200, f"{nombre} devolvio {r.status_code}"
        assert r.json()["model_loaded"] is True, f"{nombre} no tiene el modelo cargado"


def test_entornos_son_distintos():
    """
    Confirma que efectivamente se estan comparando dos arquitecturas.

    Evita el falso positivo de apuntar ambas variables a la misma URL, en cuyo
    caso la paridad se cumpliria trivialmente.
    """
    entorno_ec2 = requests.get(f"{EC2_URL}/health", timeout=TIMEOUT).json()
    entorno_lambda = requests.get(f"{LAMBDA_URL}/health", timeout=TIMEOUT).json()

    assert entorno_ec2["environment"] == "ec2"
    assert entorno_lambda["environment"] == "lambda"


@pytest.mark.parametrize("ruta,payload", CASOS)
def test_paridad_de_resultados(ruta, payload):
    """Una misma solicitud debe producir la misma respuesta en ambos despliegues."""
    respuesta_ec2 = pedir(EC2_URL, ruta, payload)
    respuesta_lambda = pedir(LAMBDA_URL, ruta, payload)

    assert respuesta_ec2.status_code == respuesta_lambda.status_code == 200
    assert respuesta_ec2.json() == respuesta_lambda.json(), (
        f"Divergencia en {ruta} con payload {payload}"
    )


@pytest.mark.parametrize("ruta,payload", INVALIDOS)
def test_paridad_ante_entradas_invalidas(ruta, payload):
    """Ambos despliegues deben rechazar las mismas entradas con codigo 4xx."""
    codigo_ec2 = pedir(EC2_URL, ruta, payload).status_code
    codigo_lambda = pedir(LAMBDA_URL, ruta, payload).status_code

    assert 400 <= codigo_ec2 < 500
    assert 400 <= codigo_lambda < 500
    assert codigo_ec2 == codigo_lambda


def test_paridad_visualizacion_dependencias():
    """
    Ambos despliegues deben generar un HTML con SVG para el mismo texto.

    No se comparan las cadenas completas: displaCy incorpora identificadores
    generados aleatoriamente en cada render, de modo que la comparacion se hace
    sobre la estructura y el contenido linguistico.
    """
    payload = {"text": "El gato come pescado."}
    html_ec2 = pedir(EC2_URL, f"{PREFIJO}/visualize/dep", payload)
    html_lambda = pedir(LAMBDA_URL, f"{PREFIJO}/visualize/dep", payload)

    assert html_ec2.status_code == html_lambda.status_code == 200
    for respuesta in (html_ec2, html_lambda):
        assert "text/html" in respuesta.headers["content-type"]
        assert "<svg" in respuesta.text

    for palabra in ("gato", "come", "pescado"):
        assert palabra in html_ec2.text
        assert palabra in html_lambda.text
