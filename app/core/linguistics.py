"""
Analisis linguistico: dependencias sintacticas y entidades nombradas.

A diferencia del preprocesamiento, estas funciones operan sobre el texto crudo.
Aplicar limpieza agresiva antes del analisis sintactico o del NER degrada los
resultados: el parser depende de la puntuacion para delimitar clausulas y el
reconocedor de entidades usa las mayusculas como senal principal para detectar
nombres propios.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.pipeline import get_nlp

# Descripciones legibles de las etiquetas de entidad del modelo espanol.
ETIQUETAS_ENTIDAD = {
    "PER": "Persona",
    "PERSON": "Persona",
    "ORG": "Organizacion",
    "LOC": "Lugar",
    "GPE": "Entidad geopolitica",
    "MISC": "Miscelanea",
    "DATE": "Fecha",
    "TIME": "Hora",
    "MONEY": "Cantidad monetaria",
    "PERCENT": "Porcentaje",
}


def analyze_dependencies(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Devuelve el arbol de dependencias sintacticas del texto.

    Para cada token se reporta su relacion de dependencia (dep_), su nucleo
    sintactico (head) y sus hijos, junto con la etiqueta morfosintactica. Se
    identifica ademas la raiz de cada oracion, que es el nodo del cual cuelga
    toda la estructura.
    """
    nlp = get_nlp(model)
    doc = nlp(text)

    tokens: List[Dict[str, Any]] = []
    for token in doc:
        if token.is_space:
            continue
        tokens.append(
            {
                "indice": token.i,
                "texto": token.text,
                "lema": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dependencia": token.dep_,
                "explicacion_dependencia": spacy_explain(token.dep_),
                "nucleo": token.head.text,
                "indice_nucleo": token.head.i,
                "hijos": [hijo.text for hijo in token.children],
                "es_raiz": token.dep_ == "ROOT",
            }
        )

    oraciones = [
        {
            "texto": sent.text.strip(),
            "raiz": sent.root.text,
            "pos_raiz": sent.root.pos_,
        }
        for sent in doc.sents
    ]

    return {
        "texto": text,
        "tokens": tokens,
        "oraciones": oraciones,
        "num_tokens": len(tokens),
        "num_oraciones": len(oraciones),
    }


def spacy_explain(label: str) -> str:
    """
    Traduce una etiqueta de spaCy a su descripcion legible.

    Se envuelve en una funcion propia porque spacy.explain devuelve None para
    etiquetas desconocidas, lo que ensucia la respuesta JSON.
    """
    import spacy

    return spacy.explain(label) or label


def analyze_entities(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Reconoce entidades nombradas y devuelve el texto anotado.

    Ademas de la lista estructurada de entidades con sus offsets, se construye
    una version del texto con las etiquetas insertadas en linea, en el formato
    'Bogota [LOC]'. La insercion se hace recorriendo las entidades en orden
    inverso para que los offsets de las anteriores no se invaliden.
    """
    nlp = get_nlp(model)
    doc = nlp(text)

    entidades: List[Dict[str, Any]] = []
    for ent in doc.ents:
        entidades.append(
            {
                "texto": ent.text,
                "etiqueta": ent.label_,
                "descripcion": ETIQUETAS_ENTIDAD.get(
                    ent.label_, spacy_explain(ent.label_)
                ),
                "inicio": ent.start_char,
                "fin": ent.end_char,
                "token_inicio": ent.start,
                "token_fin": ent.end,
            }
        )

    texto_etiquetado = text
    for ent in reversed(doc.ents):
        texto_etiquetado = (
            texto_etiquetado[: ent.end_char]
            + f" [{ent.label_}]"
            + texto_etiquetado[ent.end_char :]
        )

    conteo: Dict[str, int] = {}
    for ent in entidades:
        conteo[ent["etiqueta"]] = conteo.get(ent["etiqueta"], 0) + 1

    # Esquema BIO, util si posteriormente se entrena un modelo de secuencias.
    bio = [
        {"token": token.text, "bio": f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else "O"}
        for token in doc
        if not token.is_space
    ]

    return {
        "texto": text,
        "texto_etiquetado": texto_etiquetado,
        "entidades": entidades,
        "conteo_por_etiqueta": conteo,
        "num_entidades": len(entidades),
        "esquema_bio": bio,
    }
