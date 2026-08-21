"""
Analisis linguistico: POS, entidades nombradas y visualizacion de dependencias.

Estas capacidades operan sobre el texto ORIGINAL y no sobre el texto limpio.
La razon es doble y esta impuesta por el propio contrato:

  - Los indices de las entidades deben referirse al texto original, con start
    inclusivo y end exclusivo. Limpiar el texto destruiria esa correspondencia.
  - El reconocedor de entidades usa la capitalizacion como senal principal y el
    analizador de dependencias usa la puntuacion para delimitar clausulas.
    Aplicar la limpieza antes degradaria ambos resultados de forma severa.
"""

from __future__ import annotations

from typing import Any, Dict, List

from spacy import displacy

from app.core.pipeline import get_nlp


def analizar_pos(textos: List[str]) -> List[Dict[str, Any]]:
    """
    Devuelve tokens con texto, categoria gramatical universal y lema.

    La posicion i del resultado corresponde al documento i de la entrada y el
    orden de los tokens dentro de cada documento se conserva.
    """
    nlp = get_nlp()
    resultados: List[Dict[str, Any]] = []

    for doc in nlp.pipe(textos):
        tokens = [
            {"text": token.text, "pos": token.pos_, "lemma": token.lemma_}
            for token in doc
            if not token.is_space
        ]
        resultados.append({"tokens": tokens})

    return resultados


def analizar_ner(textos: List[str]) -> List[Dict[str, Any]]:
    """
    Detecta entidades nombradas con su texto, tipo y posicion.

    Los offsets start y end se toman directamente de spaCy, que los expresa en
    caracteres sobre el texto original con start inclusivo y end exclusivo,
    exactamente como exige el contrato.
    """
    nlp = get_nlp()
    resultados: List[Dict[str, Any]] = []

    for doc in nlp.pipe(textos):
        entidades = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        resultados.append({"entities": entidades})

    return resultados


def visualizar_dependencias(texto: str) -> str:
    """
    Genera un documento HTML con la representacion SVG del analisis sintactico.

    Se usa displacy.render con page=True para obtener un documento HTML
    completo y valido que contiene el SVG, tal como exige la guia. Se procesa
    un unico documento por solicitud.
    """
    nlp = get_nlp()
    doc = nlp(texto)
    return displacy.render(doc, style="dep", page=True, options={"compact": True})
