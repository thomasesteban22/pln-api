"""
Carga y gestion del pipeline de spaCy.

El modelo se carga de forma perezosa (lazy) y se mantiene como singleton a nivel
de modulo. Esto es deliberado: en AWS Lambda el contenedor se reutiliza entre
invocaciones, de modo que el costo de cargar el modelo (~1 s) se paga una sola
vez en el arranque en frio y no en cada peticion.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import spacy
from spacy.language import Language

# Modelo por defecto. Puede sobreescribirse por variable de entorno para
# desplegar una version distinta sin tocar el codigo.
DEFAULT_MODEL = os.environ.get("SPACY_MODEL", "es_core_news_sm")


@lru_cache(maxsize=2)
def get_nlp(model_name: Optional[str] = None) -> Language:
    """
    Devuelve el objeto Language de spaCy correspondiente al modelo solicitado.

    El decorador lru_cache garantiza que el modelo se cargue una unica vez por
    proceso. Si el modelo no esta instalado se lanza un OSError con un mensaje
    accionable en lugar del error generico de spaCy.
    """
    name = model_name or DEFAULT_MODEL
    try:
        return spacy.load(name)
    except OSError as exc:
        raise OSError(
            f"El modelo de spaCy '{name}' no esta instalado. "
            f"Instalelo con el wheel del release oficial, por ejemplo:\n"
            f"  pip install https://github.com/explosion/spacy-models/releases/"
            f"download/{name}-3.8.0/{name}-3.8.0-py3-none-any.whl"
        ) from exc


def warmup() -> None:
    """
    Fuerza la carga del modelo y ejecuta una inferencia trivial.

    Se invoca en el arranque de la aplicacion para que la primera peticion real
    del usuario no absorba el costo de inicializacion.
    """
    nlp = get_nlp()
    nlp("Texto de calentamiento.")
