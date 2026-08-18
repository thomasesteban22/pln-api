"""
Pruebas contra la funcion Lambda real, invocada directamente por la API de AWS.

Este modulo cumple el mismo proposito que test_remoto.py pero sin depender de
una Function URL publica. Se usa cuando la URL no esta disponible, por ejemplo
en cuentas con restricciones de organizacion que impiden exponer funciones sin
autenticacion.

La invocacion directa ejercita exactamente el mismo camino de codigo que una
peticion HTTP real: AWS entrega el evento a Mangum, que lo traduce a una
peticion ASGI y se la pasa a FastAPI. La unica diferencia es el transporte.

Uso:
    export LAMBDA_FUNCTION=pln-api
    export AWS_REGION=us-east-1
    pytest tests/test_lambda.py -v

Si LAMBDA_FUNCTION no esta definida, las pruebas se omiten en lugar de fallar.
"""

from __future__ import annotations

import json
import os
import time

import pytest

boto3 = pytest.importorskip("boto3", reason="boto3 no esta instalado.")

FUNCION = os.environ.get("LAMBDA_FUNCTION", "")
REGION = os.environ.get("AWS_REGION", "us-east-1")

pytestmark = pytest.mark.skipif(
    not FUNCION,
    reason="Defina la variable de entorno LAMBDA_FUNCTION para ejecutar.",
)

CORPUS = [
    "gato gato gato comer pescado",
    "juan comer bogota",
    "caballo comer rapido",
]
TEXTO = "Juan Pérez viajó a Bogotá para reunirse con Ecopetrol."


@pytest.fixture(scope="session")
def cliente():
    """Cliente de Lambda con tiempos de espera holgados para el arranque en frio."""
    from botocore.config import Config

    return boto3.client(
        "lambda",
        region_name=REGION,
        config=Config(read_timeout=90, connect_timeout=15, retries={"max_attempts": 0}),
    )


def construir_evento(metodo: str, ruta: str, cuerpo: dict | None = None) -> dict:
    """
    Construye un evento con el formato de payload 2.0 de Function URL.

    El campo 'version' es obligatorio: Mangum lo usa para deducir cual de los
    formatos soportados (API Gateway REST, HTTP API, ALB, Lambda@Edge)
    corresponde al evento recibido.
    """
    evento = {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": ruta,
        "rawQueryString": "",
        "headers": {"content-type": "application/json"},
        "requestContext": {
            "http": {
                "method": metodo,
                "path": ruta,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "pytest",
            },
            "requestId": "test",
            "stage": "$default",
        },
        "isBase64Encoded": False,
    }
    if cuerpo is not None:
        evento["body"] = json.dumps(cuerpo, ensure_ascii=False)
    return evento


def invocar(cliente, metodo: str, ruta: str, cuerpo: dict | None = None) -> tuple:
    """
    Invoca la funcion y devuelve el par (codigo de estado, cuerpo deserializado).

    Se verifica que la invocacion no haya producido un error de funcion, que es
    distinto de un codigo HTTP de error: FunctionError indica una excepcion no
    controlada dentro del handler.
    """
    respuesta = cliente.invoke(
        FunctionName=FUNCION,
        Payload=json.dumps(construir_evento(metodo, ruta, cuerpo)),
    )

    assert "FunctionError" not in respuesta, (
        f"La funcion lanzo una excepcion: "
        f"{respuesta['Payload'].read().decode('utf-8')[:500]}"
    )

    resultado = json.loads(respuesta["Payload"].read().decode("utf-8"))
    codigo = resultado.get("statusCode")
    contenido = json.loads(resultado["body"]) if "body" in resultado else {}
    return codigo, contenido


# ---------------------------------------------------------------------------
# Disponibilidad
# ---------------------------------------------------------------------------


def test_funcion_responde(cliente):
    """El health check debe confirmar que se ejecuta en Lambda con el modelo cargado."""
    codigo, cuerpo = invocar(cliente, "GET", "/health")
    assert codigo == 200
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["entorno"] == "lambda"
    assert cuerpo["modelo"] == "es_core_news_sm"


def test_catalogo_de_endpoints(cliente):
    """La raiz debe describir los cinco endpoints de procesamiento."""
    codigo, cuerpo = invocar(cliente, "GET", "/")
    assert codigo == 200
    for ruta in ["POST /processed", "POST /dependency", "POST /ner",
                 "POST /full", "POST /encoding"]:
        assert ruta in cuerpo["endpoints"]


# ---------------------------------------------------------------------------
# Endpoints de procesamiento
# ---------------------------------------------------------------------------


def test_processed(cliente):
    """
    El preprocesamiento debe eliminar ruido y lematizar.

    El texto de prueba es una oracion completa de forma deliberada. El
    lematizador de spaCy es sensible al contexto sintactico: con 'los gatos
    corrían' aislado, el modelo es_core_news_sm devuelve el lema invalido
    'corer', mientras que en una oracion con complemento devuelve 'correr'.
    Ver test_lematizacion_depende_del_contexto para el caso documentado.
    """
    codigo, cuerpo = invocar(
        cliente,
        "POST",
        "/processed",
        {"text": "<p>Los gatos corrían rápidamente por los tejados</p> 😀"},
    )
    assert codigo == 200
    assert "<p>" not in cuerpo["limpieza"]["texto_limpio"]
    assert "😀" not in cuerpo["limpieza"]["texto_limpio"]
    assert "correr" in cuerpo["transformacion"]["tokens_procesados"]


def test_lematizacion_depende_del_contexto(cliente):
    """
    Documenta una limitacion del modelo es_core_news_sm.

    La misma forma verbal produce lemas distintos segun el contexto: en una
    oracion completa se obtiene 'correr', pero al recortar el complemento el
    modelo devuelve 'corer', que no es una palabra valida del espanol. El
    modelo asigna la etiqueta VERB correctamente en ambos casos; el fallo esta
    en el componente de lematizacion.

    Esto matiza la afirmacion habitual de que la lematizacion garantiza formas
    validas del idioma, y constituye un argumento medible a favor de usar un
    modelo mayor (md o lg) cuando la calidad del lema es critica.
    """
    completo = {"text": "Los gatos corrían rápidamente por los tejados"}
    recortado = {"text": "Los gatos corrían"}

    codigo_a, cuerpo_a = invocar(cliente, "POST", "/processed", completo)
    codigo_b, cuerpo_b = invocar(cliente, "POST", "/processed", recortado)

    assert codigo_a == 200 and codigo_b == 200

    lemas_a = cuerpo_a["transformacion"]["tokens_procesados"]
    lemas_b = cuerpo_b["transformacion"]["tokens_procesados"]

    assert "correr" in lemas_a, "Con contexto suficiente el lema debe ser correcto."
    assert "correr" not in lemas_b, (
        "Comportamiento del modelo cambiado: el caso degenerado ya no se "
        "reproduce. Revisar si se actualizo la version del modelo."
    )


def test_processed_documenta_transformaciones(cliente):
    """Cada token descartado debe indicar el motivo de su exclusion."""
    codigo, cuerpo = invocar(cliente, "POST", "/processed", {"text": TEXTO})
    assert codigo == 200
    motivos = {d["motivo"] for d in cuerpo["transformacion"]["tokens_descartados"]}
    assert motivos & {"stop_word", "puntuacion"}


def test_dependency(cliente):
    """Debe existir exactamente una raiz sintactica por oracion."""
    codigo, cuerpo = invocar(
        cliente, "POST", "/dependency", {"text": "El gato come pescado."}
    )
    assert codigo == 200
    raices = [t for t in cuerpo["tokens"] if t["es_raiz"]]
    assert len(raices) == 1
    assert raices[0]["pos"] == "VERB"


def test_ner(cliente):
    """Deben reconocerse entidades y devolverse el texto etiquetado."""
    codigo, cuerpo = invocar(cliente, "POST", "/ner", {"text": TEXTO})
    assert codigo == 200
    assert cuerpo["num_entidades"] > 0
    for entidad in cuerpo["entidades"]:
        assert f"[{entidad['etiqueta']}]" in cuerpo["texto_etiquetado"]


def test_ner_offsets(cliente):
    """Los offsets deben recuperar exactamente el texto de cada entidad."""
    codigo, cuerpo = invocar(cliente, "POST", "/ner", {"text": TEXTO})
    assert codigo == 200
    for entidad in cuerpo["entidades"]:
        assert TEXTO[entidad["inicio"]:entidad["fin"]] == entidad["texto"]


def test_full(cliente):
    """El analisis completo debe integrar las tres secciones y el resumen."""
    codigo, cuerpo = invocar(cliente, "POST", "/full", {"text": TEXTO})
    assert codigo == 200
    for seccion in ["preprocesamiento", "dependencias", "entidades", "resumen"]:
        assert seccion in cuerpo


def test_encoding_tfidf(cliente):
    """TF-IDF debe asignar el peso minimo al termino presente en todo el corpus."""
    codigo, cuerpo = invocar(
        cliente, "POST", "/encoding", {"documents": CORPUS, "method": "tfidf"}
    )
    assert codigo == 200
    assert cuerpo["idf"]["comer"] == min(cuerpo["idf"].values())


def test_encoding_bow(cliente):
    """BoW debe reflejar la frecuencia absoluta de cada termino."""
    codigo, cuerpo = invocar(
        cliente, "POST", "/encoding", {"documents": CORPUS, "method": "bow"}
    )
    assert codigo == 200
    assert cuerpo["matriz"][0][cuerpo["vocabulario"].index("gato")] == 3


def test_encoding_comparacion(cliente):
    """La comparacion debe devolver las tres codificaciones."""
    codigo, cuerpo = invocar(
        cliente, "POST", "/encoding", {"documents": CORPUS, "compare_all": True}
    )
    assert codigo == 200
    assert set(cuerpo["resultados"].keys()) == {"one_hot", "bow", "tfidf"}


# ---------------------------------------------------------------------------
# Validacion y rendimiento
# ---------------------------------------------------------------------------


def test_validacion_de_entrada(cliente):
    """El texto vacio debe rechazarse con 422, no provocar una excepcion."""
    codigo, _ = invocar(cliente, "POST", "/processed", {"text": ""})
    assert codigo == 422


def test_metodo_de_codificacion_invalido(cliente):
    """Un metodo no soportado debe rechazarse en la validacion."""
    codigo, _ = invocar(
        cliente, "POST", "/encoding", {"documents": CORPUS, "method": "word2vec"}
    )
    assert codigo == 422


def test_latencia_en_caliente(cliente):
    """Tras el arranque en frio la funcion debe responder con rapidez."""
    invocar(cliente, "POST", "/ner", {"text": TEXTO})  # calentamiento

    inicio = time.perf_counter()
    codigo, _ = invocar(cliente, "POST", "/ner", {"text": TEXTO})
    duracion = time.perf_counter() - inicio

    print(f"\nLatencia en caliente: {duracion * 1000:.0f} ms")
    assert codigo == 200
    assert duracion < 10
