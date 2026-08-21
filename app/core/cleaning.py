"""
Limpieza y normalizacion de texto.

Implementa la capacidad "Limpieza de texto" definida en la guia del laboratorio:
convertir a minusculas, eliminar signos de puntuacion y las palabras marcadas
como stopwords por spaCy (Token.is_stop), y normalizar los espacios en blanco,
conservando letras acentuadas, la letra ñ y los digitos.

Decision de implementacion relevante: la puntuacion se sustituye por espacios
ANTES de tokenizar. El tokenizador de spaCy separa 'casa,perro' en tres tokens
pero deja 'gato;pez' como uno solo, lo que produciria el termino 'gato;pez' y
violaria la regla de que los signos de puntuacion actuan como separadores y no
deben concatenar terminos. La sustitucion previa garantiza el comportamiento
exigido para cualquier signo.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import List

from app.core.pipeline import get_nlp

RE_ESPACIOS = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _tabla_puntuacion() -> dict:
    """
    Construye la tabla de traduccion que sustituye puntuacion por espacios.

    Se recorren los puntos de codigo del plano multilingue basico y se
    seleccionan los de categoria Unicode 'P' (puntuacion, en todas sus
    variantes: Pc, Pd, Pe, Pf, Pi, Po, Ps). Las letras acentuadas, la ñ y los
    digitos pertenecen a otras categorias y por tanto no se ven afectados.
    """
    return {
        cp: " "
        for cp in range(0x10000)
        if unicodedata.category(chr(cp)).startswith("P")
    }


def sustituir_puntuacion(texto: str) -> str:
    """Reemplaza cada signo de puntuacion por un espacio."""
    return texto.translate(_tabla_puntuacion())


def _es_termino_valido(texto: str) -> bool:
    """
    Determina si una cadena constituye un termino conservable.

    Se exige al menos un caracter alfanumerico para descartar residuos como
    simbolos de moneda o matematicos que sobreviven a la sustitucion de
    puntuacion por no pertenecer a la categoria Unicode 'P'.
    """
    return any(ch.isalnum() for ch in texto)


def limpiar(texto: str) -> str:
    """
    Aplica la limpieza completa a un documento y devuelve la cadena resultante.

    El texto se procesa con su capitalizacion original para que el tokenizador
    y el detector de stopwords operen sobre la forma natural; la conversion a
    minusculas se aplica al emitir cada token conservado.
    """
    nlp = get_nlp()
    preparado = RE_ESPACIOS.sub(" ", sustituir_puntuacion(texto)).strip()

    if not preparado:
        return ""

    terminos = [
        token.lower_
        for token in nlp(preparado)
        if not token.is_space
        and not token.is_punct
        and not token.is_stop
        and _es_termino_valido(token.text)
    ]
    return " ".join(terminos)


def limpiar_lote(textos: List[str]) -> List[str]:
    """
    Limpia una coleccion de documentos preservando el orden de entrada.

    Se usa nlp.pipe para procesar el lote en bloque, lo que reduce de forma
    apreciable el tiempo total frente a invocar el pipeline documento por
    documento.
    """
    nlp = get_nlp()
    preparados = [
        RE_ESPACIOS.sub(" ", sustituir_puntuacion(t)).strip() for t in textos
    ]

    resultados: List[str] = []
    for doc in nlp.pipe(preparados):
        terminos = [
            token.lower_
            for token in doc
            if not token.is_space
            and not token.is_punct
            and not token.is_stop
            and _es_termino_valido(token.text)
        ]
        resultados.append(" ".join(terminos))
    return resultados


def terminos(texto_limpio: str) -> List[str]:
    """Divide un texto ya limpio en la lista de terminos que lo componen."""
    return texto_limpio.split() if texto_limpio else []
