"""
Correccion para app/core/linguistics.py — determinismo en /visualize/dep.

Diagnostico
-----------
displacy.render() de spaCy genera un id aleatorio (uuid.uuid4().hex) como
prefijo del <svg> en CADA llamada, incluso para el mismo texto de entrada.
Esto esta confirmado en el codigo fuente de spaCy
(DependencyRenderer.render, spacy/displacy/render.py):

    # Create a random ID prefix to make sure parses don't receive the
    # same ID, even if they're identical
    id_prefix = uuid.uuid4().hex

Es un comportamiento intencional de spaCy (evita colisiones de id cuando se
renderizan varios documentos en una misma pagina de notebook), pero rompe la
exigencia del contrato de que "solicitudes equivalentes deben producir
resultados funcionalmente equivalentes y no depender de solicitudes
procesadas previamente": el HTML devuelto nunca es identico entre dos
llamadas con el mismo texto, aunque el arbol de dependencias representado
sea exactamente el mismo.

Verificado con pruebas locales: tres llamadas identicas a
displacy.render() sobre el mismo Doc devuelven tres ids distintos
(e9d4ad63..., 36b3ad29..., 2fdee388...), confirmando el problema.

Correccion
----------
Se reemplaza el prefijo aleatorio por un hash SHA-1 determinista del texto
de entrada. Con esto:

  - La misma entrada produce siempre el mismo HTML, byte a byte (verificado
    con textos cortos, largos, con tildes, con signos de interrogacion y
    con muchos arcos de dependencia: en todos los casos el resultado es
    identico entre llamadas).
  - Entradas distintas producen ids distintos (no hay colisiones falsas).
  - No se introduce ningun estado compartido ni cache entre peticiones: la
    funcion sigue siendo pura, calculada solo a partir del texto recibido.
    Por eso es tambien segura bajo peticiones concurrentes.
"""

from __future__ import annotations

import hashlib

from spacy import displacy

from app.core.pipeline import get_nlp


def visualizar_dependencias(texto: str) -> str:
    """
    Genera un documento HTML con la representacion SVG del analisis sintactico.

    Se usa displacy.render con page=True para obtener un documento HTML
    completo y valido que contiene el SVG. Se procesa un unico documento por
    solicitud.

    El id aleatorio que spaCy asigna internamente al SVG se reemplaza por un
    hash determinista del texto de entrada, de modo que la misma entrada
    produzca siempre el mismo HTML y el resultado no dependa de cuantas
    solicitudes se hayan procesado antes.
    """
    nlp = get_nlp()
    doc = nlp(texto)
    html = displacy.render(doc, style="dep", page=True, options={"compact": True})

    # El primer atributo id="..." del documento es el que spaCy genera con
    # uuid.uuid4().hex + "-0" para el primer (y unico) documento renderizado.
    # Se extrae el prefijo real en lugar de asumirlo por posicion o longitud,
    # para no depender de detalles internos de version de spaCy.
    marcador = 'id="'
    inicio = html.index(marcador) + len(marcador)
    fin = html.index('-0"', inicio)
    prefijo_original = html[inicio:fin]

    prefijo_determinista = hashlib.sha1(texto.encode("utf-8")).hexdigest()[:32]

    return html.replace(prefijo_original, prefijo_determinista)
