"""
Pruebas de caja negra sobre el contrato del Laboratorio I.

Replican la evaluacion automatizada descrita en la seccion 7 de la guia:
funcionalidades, contrato HTTP, respuesta ante entradas invalidas, procesamiento
por lotes, concurrencia y rendimiento.

Ejecucion contra la aplicacion en memoria:
    pytest tests/test_contrato.py -v

Ejecucion contra un despliegue real:
    API_URL=http://<ip>:8000 pytest tests/test_contrato.py -v
    API_URL=https://<id>.lambda-url.us-east-1.on.aws pytest tests/test_contrato.py -v

El mismo archivo sirve para los tres escenarios: si API_URL esta definida se
usan peticiones HTTP reales; si no, se usa TestClient sobre la aplicacion.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

PREFIJO = "/api/v1"
API_URL = os.environ.get("API_URL", "").rstrip("/")

TEXTO = "Juan Pérez viajó a Bogotá para reunirse con Ecopetrol."
CORPUS = [
    "El gato come pescado y el gato duerme.",
    "Juan come en Bogotá.",
    "El caballo come rápido.",
]


# ---------------------------------------------------------------------------
# Cliente unificado
# ---------------------------------------------------------------------------


class ClienteHTTP:
    """Cliente contra un despliegue real."""

    def __init__(self, base: str):
        import requests

        self.base = base
        self.sesion = requests.Session()

    def post(self, ruta: str, payload):
        return self.sesion.post(f"{self.base}{ruta}", json=payload, timeout=60)

    def get(self, ruta: str):
        return self.sesion.get(f"{self.base}{ruta}", timeout=60)


class ClienteLocal:
    """
    Cliente contra la aplicacion en memoria.

    Envuelve TestClient para exponer la misma firma que ClienteHTTP, de modo que
    las pruebas se escriban una sola vez y sirvan tanto para la verificacion
    local como para la de cada despliegue.
    """

    def __init__(self):
        from fastapi.testclient import TestClient

        from app.api import app

        self.cliente = TestClient(app)

    def post(self, ruta: str, payload):
        return self.cliente.post(ruta, json=payload)

    def get(self, ruta: str):
        return self.cliente.get(ruta)


@pytest.fixture(scope="session")
def cliente():
    """Devuelve el cliente adecuado segun exista o no la variable API_URL."""
    return ClienteHTTP(API_URL) if API_URL else ClienteLocal()


# ---------------------------------------------------------------------------
# Limpieza de texto
# ---------------------------------------------------------------------------


def test_clean_devuelve_lista_para_entrada_individual(cliente):
    """cleaned_text debe ser lista incluso cuando la entrada es un unico string."""
    r = cliente.post(f"{PREFIJO}/clean", {"text": "El gato come pescado."})
    assert r.status_code == 200
    cuerpo = r.json()
    assert isinstance(cuerpo["cleaned_text"], list)
    assert len(cuerpo["cleaned_text"]) == 1


def test_clean_minusculas_y_sin_stopwords(cliente):
    """El resultado debe estar en minusculas y sin palabras vacias."""
    r = cliente.post(f"{PREFIJO}/clean", {"text": "El gato come pescado."})
    limpio = r.json()["cleaned_text"][0]
    assert limpio == limpio.lower()
    assert " el " not in f" {limpio} "


def test_clean_conserva_acentos_enie_y_digitos(cliente):
    """Tildes, ñ y digitos deben sobrevivir a la limpieza."""
    r = cliente.post(
        f"{PREFIJO}/clean", {"text": "Compré 25 kilos de ñame en Bogotá."}
    )
    limpio = r.json()["cleaned_text"][0]
    assert "25" in limpio
    assert "ñame" in limpio
    assert "bogotá" in limpio


def test_clean_puntuacion_no_concatena_terminos(cliente):
    """
    Los signos de puntuacion deben actuar como separadores.

    Se incluye 'gato;pez' de forma deliberada: el tokenizador de spaCy no
    separa esa construccion por si solo, de modo que la prueba verifica la
    normalizacion previa de puntuacion.
    """
    r = cliente.post(f"{PREFIJO}/clean", {"text": "casa,perro. gato;pez"})
    limpio = r.json()["cleaned_text"][0]
    for termino in ["casa", "perro", "gato", "pez"]:
        assert termino in limpio.split()
    assert ";" not in limpio
    assert "," not in limpio


def test_clean_normaliza_espacios(cliente):
    """No deben quedar espacios dobles ni sobrantes en los extremos."""
    r = cliente.post(f"{PREFIJO}/clean", {"text": "El   gato    come.  "})
    limpio = r.json()["cleaned_text"][0]
    assert "  " not in limpio
    assert limpio == limpio.strip()


def test_clean_batch_conserva_orden(cliente):
    """La posicion i del resultado debe corresponder al documento i."""
    documentos = ["El gato come.", "Juan vive en Bogotá.", "El caballo corre."]
    r = cliente.post(f"{PREFIJO}/clean", {"text": documentos})
    limpios = r.json()["cleaned_text"]
    assert len(limpios) == 3
    assert "gato" in limpios[0]
    assert "bogotá" in limpios[1]
    assert "caballo" in limpios[2]


# ---------------------------------------------------------------------------
# POS
# ---------------------------------------------------------------------------


def test_pos_estructura(cliente):
    """Cada token debe traer text, pos y lemma."""
    r = cliente.post(f"{PREFIJO}/pos", {"text": "El gato come pescado."})
    assert r.status_code == 200
    tokens = r.json()["results"][0]["tokens"]
    assert len(tokens) > 0
    for campo in ("text", "pos", "lemma"):
        assert campo in tokens[0]


def test_pos_conserva_orden_de_tokens(cliente):
    """El orden de los tokens debe reproducir el del texto original."""
    r = cliente.post(f"{PREFIJO}/pos", {"text": "El gato come pescado"})
    textos = [t["text"] for t in r.json()["results"][0]["tokens"]]
    assert textos == ["El", "gato", "come", "pescado"]


def test_pos_lematiza(cliente):
    """Los lemas deben reducir las formas flexionadas."""
    r = cliente.post(
        f"{PREFIJO}/pos", {"text": "Los gatos corrían rápidamente por los tejados"}
    )
    lemas = [t["lemma"] for t in r.json()["results"][0]["tokens"]]
    assert "correr" in lemas
    assert "gato" in lemas


def test_pos_batch_correspondencia(cliente):
    """results[i] debe corresponder al documento i de entrada."""
    r = cliente.post(f"{PREFIJO}/pos", {"text": ["El gato", "Juan corre"]})
    resultados = r.json()["results"]
    assert len(resultados) == 2
    assert resultados[0]["tokens"][1]["text"] == "gato"
    assert resultados[1]["tokens"][0]["text"] == "Juan"


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------


def test_ner_estructura(cliente):
    """Cada entidad debe traer text, label, start y end."""
    r = cliente.post(f"{PREFIJO}/ner", {"text": TEXTO})
    assert r.status_code == 200
    entidades = r.json()["results"][0]["entities"]
    assert len(entidades) > 0
    for campo in ("text", "label", "start", "end"):
        assert campo in entidades[0]


def test_ner_offsets_sobre_texto_original(cliente):
    """
    Los indices deben recuperar exactamente la entidad del texto original.

    start es inclusivo y end exclusivo, de modo que texto[start:end] debe
    coincidir con el campo text de la entidad.
    """
    r = cliente.post(f"{PREFIJO}/ner", {"text": TEXTO})
    for entidad in r.json()["results"][0]["entities"]:
        assert TEXTO[entidad["start"] : entidad["end"]] == entidad["text"]


def test_ner_sin_entidades_devuelve_lista_vacia(cliente):
    """Un texto sin nombres propios no debe fallar."""
    r = cliente.post(f"{PREFIJO}/ner", {"text": "el gato come pescado"})
    assert r.status_code == 200
    assert r.json()["results"][0]["entities"] == []


def test_ner_batch_correspondencia(cliente):
    """Cada documento debe recibir sus propias entidades, en orden."""
    r = cliente.post(
        f"{PREFIJO}/ner", {"text": ["Juan vive en Bogotá.", "el gato come"]}
    )
    resultados = r.json()["results"]
    assert len(resultados) == 2
    assert len(resultados[0]["entities"]) > 0
    assert resultados[1]["entities"] == []


# ---------------------------------------------------------------------------
# Visualizacion de dependencias
# ---------------------------------------------------------------------------


def test_visualize_dep_devuelve_html_con_svg(cliente):
    """La respuesta debe ser text/html y contener un SVG."""
    r = cliente.post(f"{PREFIJO}/visualize/dep", {"text": "El gato come pescado."})
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<svg" in r.text
    assert "</svg>" in r.text


def test_visualize_dep_rechaza_batch(cliente):
    """El uso de lotes en este endpoint esta fuera del contrato."""
    r = cliente.post(f"{PREFIJO}/visualize/dep", {"text": ["uno", "dos"]})
    assert 400 <= r.status_code < 500


# ---------------------------------------------------------------------------
# Vectorizacion
# ---------------------------------------------------------------------------


def test_vectorize_estructura(cliente):
    """La respuesta debe traer las cuatro claves del contrato."""
    r = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS})
    assert r.status_code == 200
    cuerpo = r.json()
    for clave in ("vocabulary", "one_hot", "bag_of_words", "tf_idf"):
        assert clave in cuerpo


def test_vectorize_vocabulario_lexicografico(cliente):
    """El vocabulario debe estar en orden lexicografico ascendente."""
    vocabulario = cliente.post(
        f"{PREFIJO}/vectorize", {"documents": CORPUS}
    ).json()["vocabulary"]
    assert vocabulario == sorted(vocabulario)


def test_vectorize_dimensiones(cliente):
    """bag_of_words y tf_idf deben tener dimension N x |V|."""
    cuerpo = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS}).json()
    n = len(CORPUS)
    v = len(cuerpo["vocabulary"])

    assert len(cuerpo["bag_of_words"]) == n
    assert len(cuerpo["tf_idf"]) == n
    for fila in cuerpo["bag_of_words"]:
        assert len(fila) == v
    for fila in cuerpo["tf_idf"]:
        assert len(fila) == v


def test_vectorize_bow_frecuencia_absoluta(cliente):
    """'gato' aparece dos veces en el primer documento."""
    cuerpo = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS}).json()
    j = cuerpo["vocabulary"].index("gato")
    assert cuerpo["bag_of_words"][0][j] == 2


def test_vectorize_one_hot_es_lista_de_matrices(cliente):
    """
    one_hot debe contener una matriz por documento, con |V| columnas y una fila
    por ocurrencia retenida, cada una con un unico valor 1.
    """
    cuerpo = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS}).json()
    v = len(cuerpo["vocabulary"])

    assert len(cuerpo["one_hot"]) == len(CORPUS)
    for matriz in cuerpo["one_hot"]:
        for fila in matriz:
            assert len(fila) == v
            assert sum(fila) == 1
            assert set(fila) <= {0, 1}


def test_vectorize_one_hot_filas_igualan_ocurrencias(cliente):
    """El numero de filas debe coincidir con los terminos del texto limpio."""
    payload = {"documents": CORPUS}
    cuerpo = cliente.post(f"{PREFIJO}/vectorize", payload).json()
    limpios = cliente.post(f"{PREFIJO}/clean", {"text": CORPUS}).json()["cleaned_text"]

    for matriz, limpio in zip(cuerpo["one_hot"], limpios):
        assert len(matriz) == len(limpio.split())


def test_vectorize_tfidf_formula(cliente):
    """
    Verifica tf x idf con idf(t) = ln((|D|+1)/(n_t+1)) + 1, sin normalizar.

    Se comprueba sobre 'gato', que aparece dos veces en un unico documento de
    los tres, de modo que el valor esperado es 2 x (ln(4/2) + 1).
    """
    cuerpo = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS}).json()
    j = cuerpo["vocabulary"].index("gato")

    esperado = round(2 * (math.log((3 + 1) / (1 + 1)) + 1), 4)
    assert cuerpo["tf_idf"][0][j] == esperado


def test_vectorize_tfidf_termino_universal(cliente):
    """Un termino presente en todos los documentos recibe idf igual a 1."""
    cuerpo = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS}).json()
    j = cuerpo["vocabulary"].index("come")
    for fila, documento in zip(cuerpo["tf_idf"], cuerpo["bag_of_words"]):
        assert fila[j] == round(documento[j] * 1.0, 4)


def test_vectorize_tfidf_sin_normalizar(cliente):
    """Las filas no deben tener norma L2 unitaria."""
    cuerpo = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS}).json()
    normas = [sum(v * v for v in fila) ** 0.5 for fila in cuerpo["tf_idf"]]
    assert not all(abs(n - 1.0) < 1e-6 for n in normas)


def test_vectorize_cuatro_decimales(cliente):
    """Los valores de tf_idf deben venir redondeados a cuatro decimales."""
    cuerpo = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS}).json()
    for fila in cuerpo["tf_idf"]:
        for valor in fila:
            assert round(valor, 4) == valor


def test_vectorize_orden_de_filas(cliente):
    """Las filas deben conservar el orden de los documentos recibidos."""
    invertido = list(reversed(CORPUS))
    directo = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS}).json()
    inverso = cliente.post(f"{PREFIJO}/vectorize", {"documents": invertido}).json()

    assert directo["vocabulary"] == inverso["vocabulary"]
    assert directo["bag_of_words"][0] == inverso["bag_of_words"][-1]


def test_vectorize_idf_coincide_con_formula_de_la_guia(cliente):
    """
    Verifica termino a termino que el IDF aplicado es el de la guia.

    Se reconstruye idf(t) = ln((|D|+1)/(n_t+1)) + 1 a partir del texto limpio
    devuelto por el propio servicio y se compara con el cociente entre tf_idf y
    la frecuencia absoluta. Esta prueba es la que respalda la afirmacion de que
    los parametros elegidos de scikit-learn (smooth_idf=True, norm=None)
    reproducen exactamente la formula exigida.
    """
    cuerpo = cliente.post(f"{PREFIJO}/vectorize", {"documents": CORPUS}).json()
    limpios = cliente.post(f"{PREFIJO}/clean", {"text": CORPUS}).json()["cleaned_text"]

    conjuntos = [set(texto.split()) for texto in limpios]
    total = len(CORPUS)

    for j, termino in enumerate(cuerpo["vocabulary"]):
        n_t = sum(1 for conjunto in conjuntos if termino in conjunto)
        idf_esperado = math.log((total + 1) / (n_t + 1)) + 1

        for fila_tfidf, fila_bow in zip(cuerpo["tf_idf"], cuerpo["bag_of_words"]):
            if fila_bow[j] == 0:
                assert fila_tfidf[j] == 0.0
            else:
                esperado = round(fila_bow[j] * idf_esperado, 4)
                assert fila_tfidf[j] == esperado, (
                    f"Divergencia en el termino '{termino}'"
                )


def test_vectorize_documentos_sin_terminos_utiles(cliente):
    """
    Documentos compuestos solo por stopwords no deben provocar un error 5xx.

    Tras la limpieza el vocabulario queda vacio, caso en el que scikit-learn
    lanzaria una excepcion. El servicio debe devolver estructuras vacias
    coherentes conservando la correspondencia con la entrada.
    """
    r = cliente.post(f"{PREFIJO}/vectorize", {"documents": ["el la los", "de y con"]})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["vocabulary"] == []
    assert len(cuerpo["bag_of_words"]) == 2
    assert len(cuerpo["one_hot"]) == 2


# ---------------------------------------------------------------------------
# Entradas invalidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ruta,payload,caso",
    [
        (f"{PREFIJO}/clean", {}, "campo ausente"),
        (f"{PREFIJO}/clean", {"text": None}, "valor null"),
        (f"{PREFIJO}/clean", {"text": 123}, "tipo incorrecto"),
        (f"{PREFIJO}/clean", {"text": []}, "lista vacia"),
        (f"{PREFIJO}/clean", {"text": [1, 2]}, "elementos no string"),
        (f"{PREFIJO}/clean", {"text": ""}, "texto vacio"),
        (f"{PREFIJO}/clean", {"text": "   "}, "solo espacios"),
        (f"{PREFIJO}/pos", {"text": ""}, "texto vacio en pos"),
        (f"{PREFIJO}/ner", {"text": []}, "lista vacia en ner"),
        (f"{PREFIJO}/visualize/dep", {"text": ["a", "b"]}, "batch en dep"),
        (f"{PREFIJO}/vectorize", {"documents": ["uno"]}, "menos de dos documentos"),
        (f"{PREFIJO}/vectorize", {"documents": []}, "coleccion vacia"),
        (f"{PREFIJO}/vectorize", {"documents": ["ok", ""]}, "documento vacio"),
    ],
)
def test_entradas_invalidas_producen_4xx(cliente, ruta, payload, caso):
    """Toda entrada fuera de contrato debe producir 4xx y nunca 5xx."""
    r = cliente.post(ruta, payload)
    assert 400 <= r.status_code < 500, f"{caso}: se obtuvo {r.status_code}"


def test_lote_con_elemento_invalido_se_rechaza_completo(cliente):
    """Un lote con un elemento invalido no debe producir resultados parciales."""
    r = cliente.post(f"{PREFIJO}/clean", {"text": ["texto valido", "   "]})
    assert 400 <= r.status_code < 500
    assert "cleaned_text" not in r.text


# ---------------------------------------------------------------------------
# Consistencia, capacidad, concurrencia y rendimiento
# ---------------------------------------------------------------------------


def test_consistencia_entre_solicitudes(cliente):
    """Solicitudes equivalentes deben producir resultados identicos."""
    payload = {"documents": CORPUS}
    primero = cliente.post(f"{PREFIJO}/vectorize", payload).json()

    cliente.post(f"{PREFIJO}/clean", {"text": "peticion intermedia distinta"})

    segundo = cliente.post(f"{PREFIJO}/vectorize", payload).json()
    assert primero == segundo


def test_capacidad_veinticinco_documentos(cliente):
    """Limpieza, POS y NER deben soportar 25 documentos de 1.000 caracteres."""
    documentos = [("El gato come pescado en Bogotá. " * 32)[:1000] for _ in range(25)]

    for ruta in (f"{PREFIJO}/clean", f"{PREFIJO}/pos", f"{PREFIJO}/ner"):
        r = cliente.post(ruta, {"text": documentos})
        assert r.status_code == 200, f"{ruta} devolvio {r.status_code}"
        cuerpo = r.json()
        salida = cuerpo.get("cleaned_text") or cuerpo.get("results")
        assert len(salida) == 25


def test_capacidad_diez_documentos_vectorizacion(cliente):
    """Vectorizacion debe soportar 10 documentos de 1.000 caracteres."""
    documentos = [
        (f"Documento {i} sobre gatos, perros y caballos en Bogotá. " * 20)[:1000]
        for i in range(10)
    ]
    r = cliente.post(f"{PREFIJO}/vectorize", {"documents": documentos})
    assert r.status_code == 200
    assert len(r.json()["bag_of_words"]) == 10


def test_concurrencia_cinco_solicitudes(cliente):
    """El servicio debe atender al menos cinco solicitudes concurrentes."""

    def peticion(i: int):
        return cliente.post(f"{PREFIJO}/clean", {"text": f"El gato numero {i} come."})

    with ThreadPoolExecutor(max_workers=5) as pool:
        respuestas = list(pool.map(peticion, range(5)))

    for r in respuestas:
        assert r.status_code == 200


def test_concurrencia_resultados_independientes(cliente):
    """Las respuestas concurrentes no deben mezclarse entre si."""
    textos = [f"documento numero {i} con gato" for i in range(5)]

    def peticion(texto: str):
        return cliente.post(f"{PREFIJO}/clean", {"text": texto}).json()[
            "cleaned_text"
        ][0]

    with ThreadPoolExecutor(max_workers=5) as pool:
        resultados = list(pool.map(peticion, textos))

    for i, resultado in enumerate(resultados):
        assert str(i) in resultado


def test_rendimiento_bajo_diez_segundos(cliente):
    """Cada solicitud valida debe completarse en menos de diez segundos."""
    cliente.post(f"{PREFIJO}/clean", {"text": "calentamiento"})

    casos = [
        (f"{PREFIJO}/clean", {"text": CORPUS}),
        (f"{PREFIJO}/pos", {"text": CORPUS}),
        (f"{PREFIJO}/ner", {"text": CORPUS}),
        (f"{PREFIJO}/vectorize", {"documents": CORPUS}),
        (f"{PREFIJO}/visualize/dep", {"text": TEXTO}),
    ]

    for ruta, payload in casos:
        inicio = time.perf_counter()
        r = cliente.post(ruta, payload)
        duracion = time.perf_counter() - inicio
        assert r.status_code == 200
        assert duracion < 10, f"{ruta} tardo {duracion:.2f} s"
