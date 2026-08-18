"""
Limpieza y transformacion de texto.

Separa explicitamente las dos fases del preprocesamiento vistas en clase:

  1. Limpieza (text cleaning): elimina elementos ruidosos que no aportan
     informacion. Opera sobre la cadena cruda, antes de tokenizar.
  2. Transformacion (text transformation): normaliza el texto ya tokenizado
     (minusculas, stop words, lematizacion, POS) sin distorsionar el
     significado.

Cada funcion devuelve informacion suficiente para documentar el "antes y
despues", que es exactamente la evidencia que exige el sistema de evaluacion.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

from app.core.pipeline import get_nlp

# ---------------------------------------------------------------------------
# Expresiones regulares de limpieza
# ---------------------------------------------------------------------------

RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
RE_NUMBER = re.compile(r"\b\d+([.,]\d+)*\b")
RE_WHITESPACE = re.compile(r"\s+")

# Rangos Unicode de emojis y pictogramas.
RE_EMOJI = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticones
    "\U0001f300-\U0001f5ff"  # simbolos y pictogramas
    "\U0001f680-\U0001f6ff"  # transporte y mapas
    "\U0001f1e0-\U0001f1ff"  # banderas
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f900-\U0001f9ff"  # suplemento de pictogramas
    "\U00002600-\U000026ff"  # simbolos varios
    "\U0000fe00-\U0000fe0f"  # selectores de variacion
    "]+",
    flags=re.UNICODE,
)


def strip_accents(text: str) -> str:
    """
    Elimina tildes y diacriticos preservando la letra base.

    Se implementa via descomposicion NFD y filtrado de marcas combinantes, de
    modo que 'accion' y 'acción' colapsen al mismo token. Se ofrece como opcion
    y no como comportamiento por defecto porque en espanol la tilde puede ser
    distintiva ('publico' / 'publicó').
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def clean_text(
    text: str,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_emojis: bool = True,
    remove_numbers: bool = False,
    remove_accents: bool = False,
    lowercase: bool = True,
) -> Dict[str, Any]:
    """
    Aplica la fase de limpieza sobre el texto crudo.

    Devuelve un diccionario con el texto resultante y la traza de las
    operaciones efectivamente aplicadas, junto con el numero de caracteres
    eliminados por cada una. Esa traza es la que permite justificar las
    decisiones de preprocesamiento en el informe tecnico.
    """
    original = text
    applied: List[Dict[str, Any]] = []

    def apply(name: str, pattern: re.Pattern, current: str, repl: str = " ") -> str:
        result = pattern.sub(repl, current)
        removed = len(current) - len(result)
        applied.append({"operacion": name, "caracteres_afectados": removed})
        return result

    working = original

    if remove_html:
        working = apply("eliminar_html", RE_HTML, working)
    if remove_urls:
        working = apply("eliminar_urls", RE_URL, working)
    if remove_emails:
        working = apply("eliminar_emails", RE_EMAIL, working)
    if remove_emojis:
        working = apply("eliminar_emojis", RE_EMOJI, working)
    if remove_numbers:
        working = apply("eliminar_numeros", RE_NUMBER, working)

    if remove_accents:
        before = working
        working = strip_accents(working)
        applied.append(
            {
                "operacion": "eliminar_acentos",
                "caracteres_afectados": sum(
                    1 for a, b in zip(before, working) if a != b
                ),
            }
        )

    if lowercase:
        working = working.lower()
        applied.append({"operacion": "minusculas", "caracteres_afectados": 0})

    # La normalizacion de espacios se aplica siempre al final, porque las
    # sustituciones anteriores insertan espacios.
    working = RE_WHITESPACE.sub(" ", working).strip()
    applied.append({"operacion": "normalizar_espacios", "caracteres_afectados": 0})

    return {
        "texto_original": original,
        "texto_limpio": working,
        "operaciones_aplicadas": applied,
        "longitud_original": len(original),
        "longitud_final": len(working),
    }


def transform_text(
    text: str,
    remove_stopwords: bool = True,
    remove_punctuation: bool = True,
    lemmatize: bool = True,
    allowed_pos: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aplica la fase de transformacion sobre texto ya limpio.

    Tokeniza con spaCy y filtra segun los criterios solicitados. Cuando
    lemmatize es True el token se sustituye por su lema, garantizando que la
    forma resultante sea una palabra valida del idioma y reduciendo el tamano
    del vocabulario.

    El parametro allowed_pos permite conservar unicamente ciertas categorias
    gramaticales (por ejemplo ["NOUN", "VERB", "ADJ"]), tecnica habitual cuando
    el objetivo es modelado de topicos.
    """
    nlp = get_nlp(model)
    doc = nlp(text)

    tokens_originales: List[str] = []
    tokens_finales: List[str] = []
    descartados: List[Dict[str, str]] = []

    for token in doc:
        if token.is_space:
            continue

        tokens_originales.append(token.text)

        if remove_punctuation and (token.is_punct or token.is_quote or token.is_bracket):
            descartados.append({"token": token.text, "motivo": "puntuacion"})
            continue
        if remove_stopwords and token.is_stop:
            descartados.append({"token": token.text, "motivo": "stop_word"})
            continue
        if allowed_pos and token.pos_ not in allowed_pos:
            descartados.append({"token": token.text, "motivo": f"pos_{token.pos_}"})
            continue

        tokens_finales.append(token.lemma_ if lemmatize else token.text)

    vocabulario = sorted(set(tokens_finales))

    return {
        "tokens_originales": tokens_originales,
        "tokens_procesados": tokens_finales,
        "tokens_descartados": descartados,
        "texto_procesado": " ".join(tokens_finales),
        "vocabulario": vocabulario,
        "num_tokens_originales": len(tokens_originales),
        "num_tokens_procesados": len(tokens_finales),
        "tamano_vocabulario": len(vocabulario),
    }


def process(text: str, model: Optional[str] = None, **options: Any) -> Dict[str, Any]:
    """
    Ejecuta el preprocesamiento completo: limpieza seguida de transformacion.

    Las opciones se reparten entre ambas fases segun la firma de cada funcion,
    de modo que el endpoint pueda exponer un unico objeto de configuracion.
    """
    clean_keys = {
        "remove_html",
        "remove_urls",
        "remove_emails",
        "remove_emojis",
        "remove_numbers",
        "remove_accents",
        "lowercase",
    }
    transform_keys = {
        "remove_stopwords",
        "remove_punctuation",
        "lemmatize",
        "allowed_pos",
    }

    clean_opts = {k: v for k, v in options.items() if k in clean_keys}
    transform_opts = {k: v for k, v in options.items() if k in transform_keys}

    limpieza = clean_text(text, **clean_opts)
    transformacion = transform_text(
        limpieza["texto_limpio"], model=model, **transform_opts
    )

    return {"limpieza": limpieza, "transformacion": transformacion}
