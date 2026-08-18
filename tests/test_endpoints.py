"""
Pruebas automatizadas de los endpoints de la API.

Se ejecutan contra la aplicacion en memoria mediante TestClient, por lo que no
requieren que el servidor este levantado. Para probar un despliegue real (Cloud9
o Lambda) usar tests/test_remoto.py.

Ejecucion:
    pytest -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

TEXTO_EJEMPLO = (
    "Juan Pérez viajó a Bogotá el 5 de marzo para reunirse con Ecopetrol."
)

TEXTO_RUIDOSO = (
    "<p>¡Hola!</p> Visita https://ejemplo.com o escribe a test@correo.com "
    "para más información 😀😀 sobre los 25 productos."
)

CORPUS = [
    "gato gato gato comer pescado",
    "juan comer bogota",
    "caballo comer rapido",
]


# ---------------------------------------------------------------------------
# Endpoints generales
# ---------------------------------------------------------------------------


def test_raiz_lista_endpoints():
    """La raiz debe describir el catalogo de endpoints disponibles."""
    respuesta = client.get("/")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert "endpoints" in cuerpo
    for ruta in ["POST /processed", "POST /dependency", "POST /ner",
                 "POST /full", "POST /encoding"]:
        assert ruta in cuerpo["endpoints"]


def test_health_reporta_modelo_cargado():
    """El health check debe confirmar que el modelo de spaCy esta disponible."""
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["modelo"] == "es_core_news_sm"


def test_cabecera_de_latencia_presente():
    """El middleware debe anadir la latencia a cada respuesta."""
    respuesta = client.get("/health")
    assert "X-Process-Time-Ms" in respuesta.headers
    assert float(respuesta.headers["X-Process-Time-Ms"]) >= 0


# ---------------------------------------------------------------------------
# /processed
# ---------------------------------------------------------------------------


def test_processed_devuelve_ambas_fases():
    """La respuesta debe separar limpieza y transformacion."""
    respuesta = client.post("/processed", json={"text": TEXTO_EJEMPLO})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert "limpieza" in cuerpo
    assert "transformacion" in cuerpo
    assert cuerpo["transformacion"]["num_tokens_procesados"] > 0


def test_processed_elimina_ruido():
    """HTML, URLs, correos y emojis deben desaparecer del texto limpio."""
    respuesta = client.post("/processed", json={"text": TEXTO_RUIDOSO})
    assert respuesta.status_code == 200
    limpio = respuesta.json()["limpieza"]["texto_limpio"]
    assert "<p>" not in limpio
    assert "https://" not in limpio
    assert "@correo.com" not in limpio
    assert "😀" not in limpio


def test_processed_lematiza():
    """Los verbos conjugados deben reducirse a su forma raiz."""
    respuesta = client.post(
        "/processed",
        json={"text": "Los gatos corrían rápidamente por los tejados."},
    )
    tokens = respuesta.json()["transformacion"]["tokens_procesados"]
    assert "correr" in tokens
    assert "corrían" not in tokens


def test_processed_reduce_vocabulario():
    """La lematizacion debe producir menos tokens unicos que el texto original."""
    respuesta = client.post(
        "/processed",
        json={"text": "El gato come. Los gatos comen. La gata comía."},
    )
    cuerpo = respuesta.json()["transformacion"]
    assert cuerpo["tamano_vocabulario"] < len(set(cuerpo["tokens_originales"]))


def test_processed_registra_motivo_de_descarte():
    """Cada token descartado debe indicar por que se elimino."""
    respuesta = client.post("/processed", json={"text": TEXTO_EJEMPLO})
    descartados = respuesta.json()["transformacion"]["tokens_descartados"]
    assert len(descartados) > 0
    motivos = {d["motivo"] for d in descartados}
    assert motivos & {"stop_word", "puntuacion"}


def test_processed_respeta_opciones():
    """Desactivar la lematizacion debe conservar las formas superficiales."""
    payload = {
        "text": "Los gatos corrían rápidamente.",
        "options": {"lemmatize": False, "remove_stopwords": False},
    }
    tokens = client.post("/processed", json=payload).json()["transformacion"][
        "tokens_procesados"
    ]
    assert "corrían" in tokens


def test_processed_filtra_por_categoria_gramatical():
    """El filtro allowed_pos debe conservar solo las categorias indicadas."""
    payload = {
        "text": "El gato negro come pescado fresco en la casa.",
        "options": {"allowed_pos": ["NOUN"], "lemmatize": True},
    }
    cuerpo = client.post("/processed", json=payload).json()["transformacion"]
    assert cuerpo["num_tokens_procesados"] > 0
    assert "comer" not in cuerpo["tokens_procesados"]


def test_processed_rechaza_texto_vacio():
    """Un texto vacio debe producir un error de validacion."""
    assert client.post("/processed", json={"text": "   "}).status_code == 422


def test_processed_rechaza_peticion_sin_texto():
    """Omitir el campo obligatorio debe producir 422."""
    assert client.post("/processed", json={}).status_code == 422


# ---------------------------------------------------------------------------
# /dependency
# ---------------------------------------------------------------------------


def test_dependency_identifica_raiz():
    """Toda oracion debe tener exactamente un token marcado como raiz."""
    respuesta = client.post("/dependency", json={"text": "El gato come pescado."})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    raices = [t for t in cuerpo["tokens"] if t["es_raiz"]]
    assert len(raices) == 1
    assert raices[0]["pos"] == "VERB"


def test_dependency_estructura_de_token():
    """Cada token debe traer nucleo, dependencia y categoria gramatical."""
    cuerpo = client.post("/dependency", json={"text": TEXTO_EJEMPLO}).json()
    for campo in ["texto", "lema", "pos", "dependencia", "nucleo", "hijos"]:
        assert campo in cuerpo["tokens"][0]


def test_dependency_segmenta_oraciones():
    """Un texto con dos oraciones debe reportar dos oraciones."""
    cuerpo = client.post(
        "/dependency", json={"text": "El gato duerme. El perro corre."}
    ).json()
    assert cuerpo["num_oraciones"] == 2


def test_dependency_indices_de_nucleo_validos():
    """Todo indice de nucleo debe apuntar a un token existente."""
    cuerpo = client.post("/dependency", json={"text": TEXTO_EJEMPLO}).json()
    indices = {t["indice"] for t in cuerpo["tokens"]}
    for token in cuerpo["tokens"]:
        assert token["indice_nucleo"] in indices


# ---------------------------------------------------------------------------
# /ner
# ---------------------------------------------------------------------------


def test_ner_detecta_persona_y_lugar():
    """El modelo debe reconocer el nombre propio y la ciudad."""
    cuerpo = client.post("/ner", json={"text": TEXTO_EJEMPLO}).json()
    assert cuerpo["num_entidades"] > 0
    textos = " ".join(e["texto"] for e in cuerpo["entidades"])
    assert "Juan" in textos or "Bogotá" in textos


def test_ner_texto_etiquetado_contiene_marcas():
    """El texto anotado debe incluir las etiquetas en linea."""
    cuerpo = client.post("/ner", json={"text": TEXTO_EJEMPLO}).json()
    assert "[" in cuerpo["texto_etiquetado"]
    for entidad in cuerpo["entidades"]:
        assert f"[{entidad['etiqueta']}]" in cuerpo["texto_etiquetado"]


def test_ner_offsets_coinciden_con_el_texto():
    """Los offsets deben recuperar exactamente el texto de la entidad."""
    texto = TEXTO_EJEMPLO
    cuerpo = client.post("/ner", json={"text": texto}).json()
    for entidad in cuerpo["entidades"]:
        assert texto[entidad["inicio"]:entidad["fin"]] == entidad["texto"]


def test_ner_esquema_bio_completo():
    """El esquema BIO debe cubrir todos los tokens no vacios."""
    cuerpo = client.post("/ner", json={"text": TEXTO_EJEMPLO}).json()
    assert len(cuerpo["esquema_bio"]) > 0
    for item in cuerpo["esquema_bio"]:
        assert item["bio"] == "O" or item["bio"].startswith(("B-", "I-"))


def test_ner_sin_entidades_no_falla():
    """Un texto sin nombres propios debe devolver una lista vacia, no un error."""
    respuesta = client.post("/ner", json={"text": "el gato come pescado"})
    assert respuesta.status_code == 200
    assert respuesta.json()["num_entidades"] == 0


# ---------------------------------------------------------------------------
# /full
# ---------------------------------------------------------------------------


def test_full_integra_los_tres_analisis():
    """La respuesta debe contener las tres secciones y el resumen."""
    cuerpo = client.post("/full", json={"text": TEXTO_EJEMPLO}).json()
    for seccion in ["preprocesamiento", "dependencias", "entidades", "resumen"]:
        assert seccion in cuerpo


def test_full_es_consistente_con_endpoints_individuales():
    """Los resultados de /full deben coincidir con los endpoints separados."""
    completo = client.post("/full", json={"text": TEXTO_EJEMPLO}).json()
    ner = client.post("/ner", json={"text": TEXTO_EJEMPLO}).json()
    dep = client.post("/dependency", json={"text": TEXTO_EJEMPLO}).json()

    assert completo["entidades"]["num_entidades"] == ner["num_entidades"]
    assert completo["dependencias"]["num_tokens"] == dep["num_tokens"]


def test_full_reporta_latencia():
    """El resumen debe incluir el tiempo de proceso."""
    resumen = client.post("/full", json={"text": TEXTO_EJEMPLO}).json()["resumen"]
    assert resumen["tiempo_ms"] > 0


# ---------------------------------------------------------------------------
# /encoding
# ---------------------------------------------------------------------------


def test_encoding_bow_cuenta_frecuencias():
    """BoW debe reflejar la frecuencia absoluta: 'gato' aparece tres veces."""
    cuerpo = client.post(
        "/encoding", json={"documents": CORPUS, "method": "bow"}
    ).json()
    indice = cuerpo["vocabulario"].index("gato")
    assert cuerpo["matriz"][0][indice] == 3


def test_encoding_one_hot_es_binario():
    """One-hot solo debe contener ceros y unos."""
    cuerpo = client.post(
        "/encoding", json={"documents": CORPUS, "method": "one_hot"}
    ).json()
    for fila in cuerpo["matriz"]:
        assert set(fila) <= {0, 1}


def test_encoding_one_hot_y_bow_difieren():
    """La diferencia entre binario y frecuencia debe ser observable."""
    one_hot = client.post(
        "/encoding", json={"documents": CORPUS, "method": "one_hot"}
    ).json()
    bow = client.post(
        "/encoding", json={"documents": CORPUS, "method": "bow"}
    ).json()
    assert one_hot["matriz"] != bow["matriz"]


def test_encoding_tfidf_penaliza_terminos_frecuentes():
    """'comer' aparece en los tres documentos y debe recibir el IDF minimo."""
    cuerpo = client.post(
        "/encoding", json={"documents": CORPUS, "method": "tfidf"}
    ).json()
    idf = cuerpo["idf"]
    assert idf["comer"] == min(idf.values())
    assert idf["gato"] > idf["comer"]


def test_encoding_tfidf_normalizado():
    """Con norma L2 cada vector de documento debe tener magnitud unitaria."""
    cuerpo = client.post(
        "/encoding", json={"documents": CORPUS, "method": "tfidf"}
    ).json()
    for fila in cuerpo["matriz"]:
        norma = sum(v * v for v in fila) ** 0.5
        assert abs(norma - 1.0) < 1e-6


def test_encoding_dimensiones_correctas():
    """La matriz debe ser de tamano num_documentos x tamano_vocabulario."""
    cuerpo = client.post("/encoding", json={"documents": CORPUS}).json()
    filas, columnas = cuerpo["dimensiones_matriz"]
    assert filas == len(CORPUS)
    assert columnas == cuerpo["tamano_vocabulario"]


def test_encoding_ngramas():
    """Un rango de bigramas debe ampliar el vocabulario."""
    unigramas = client.post(
        "/encoding", json={"documents": CORPUS, "method": "bow"}
    ).json()
    bigramas = client.post(
        "/encoding",
        json={"documents": CORPUS, "method": "bow", "ngram_min": 1, "ngram_max": 2},
    ).json()
    assert bigramas["tamano_vocabulario"] > unigramas["tamano_vocabulario"]


def test_encoding_comparacion_de_metodos():
    """compare_all debe devolver las tres codificaciones."""
    cuerpo = client.post(
        "/encoding", json={"documents": CORPUS, "compare_all": True}
    ).json()
    assert set(cuerpo["resultados"].keys()) == {"one_hot", "bow", "tfidf"}


def test_encoding_con_preprocesamiento():
    """La opcion preprocess debe lematizar antes de codificar."""
    cuerpo = client.post(
        "/encoding",
        json={
            "documents": ["Los gatos corrían", "El gato corre"],
            "method": "bow",
            "preprocess": True,
        },
    ).json()
    assert "correr" in cuerpo["vocabulario"]


def test_encoding_metodo_invalido():
    """Un metodo no soportado debe rechazarse en la validacion."""
    respuesta = client.post(
        "/encoding", json={"documents": CORPUS, "method": "word2vec"}
    )
    assert respuesta.status_code == 422


def test_encoding_lista_vacia():
    """Una lista de documentos vacia debe rechazarse."""
    assert client.post("/encoding", json={"documents": []}).status_code == 422


def test_encoding_rango_ngramas_invertido():
    """ngram_max menor que ngram_min debe producir 400."""
    respuesta = client.post(
        "/encoding",
        json={"documents": CORPUS, "ngram_min": 3, "ngram_max": 1},
    )
    assert respuesta.status_code == 400


# ---------------------------------------------------------------------------
# Casos limite transversales
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ruta", ["/processed", "/dependency", "/ner", "/full"])
def test_endpoints_rechazan_texto_vacio(ruta):
    """Ningun endpoint de texto debe aceptar una cadena vacia."""
    assert client.post(ruta, json={"text": ""}).status_code == 422


@pytest.mark.parametrize("ruta", ["/processed", "/dependency", "/ner", "/full"])
def test_endpoints_aceptan_texto_unicode(ruta):
    """Tildes, enes y signos de apertura deben procesarse sin error."""
    respuesta = client.post(
        ruta, json={"text": "¿Cómo está el niño? ¡Muy bien, señor!"}
    )
    assert respuesta.status_code == 200


@pytest.mark.parametrize("ruta", ["/processed", "/dependency", "/ner", "/full"])
def test_endpoints_toleran_texto_largo(ruta):
    """Un texto de tamano considerable no debe provocar fallos."""
    texto = "El gato negro come pescado fresco en Bogotá. " * 40
    assert client.post(ruta, json={"text": texto}).status_code == 200
