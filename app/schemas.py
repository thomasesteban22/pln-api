"""
Contrato de entrada de la API y reglas de validacion.

La seccion 9 de la guia enumera los casos que quedan fuera del contrato:
ausencia de campos obligatorios, valores null, tipos distintos de los
declarados, listas vacias, elementos no string, textos vacios o compuestos
unicamente por espacios, uso de batch en /visualize/dep y colecciones con menos
de dos documentos en /vectorize.

Todos estos casos deben producir una respuesta HTTP 4xx sin resultados
parciales. La validacion se realiza en Pydantic, antes de que la peticion
alcance el pipeline de spaCy, de modo que un lote con un solo elemento invalido
se rechaza completo y no llega a procesarse ningun elemento.
"""

from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel, StrictStr, field_validator

# Limites derivados del atributo de calidad "Capacidad" de la guia. Se fijan
# holgadamente por encima de los minimos exigidos (25 documentos de 1.000
# caracteres para limpieza, POS y NER; 10 documentos para vectorizacion).
MAX_DOCUMENTOS = 100
MAX_CARACTERES = 20_000
MIN_DOCUMENTOS_VECTORIZAR = 2


def _validar_texto(valor: str, etiqueta: str = "texto") -> str:
    """Rechaza cadenas vacias o compuestas unicamente por espacios."""
    if not valor.strip():
        raise ValueError(
            f"El {etiqueta} no puede estar vacio ni contener solo espacios."
        )
    if len(valor) > MAX_CARACTERES:
        raise ValueError(
            f"El {etiqueta} excede el limite de {MAX_CARACTERES} caracteres."
        )
    return valor


class PeticionTexto(BaseModel):
    """
    Peticion para las capacidades que admiten texto individual o por lotes.

    El tipo StrictStr es deliberado: sin el, Pydantic convertiria valores
    numericos o booleanos a cadena, aceptando entradas que el contrato declara
    invalidas por ser de un tipo distinto al declarado.
    """

    text: Union[StrictStr, List[StrictStr]]

    @field_validator("text")
    @classmethod
    def validar(cls, valor):
        if isinstance(valor, str):
            return _validar_texto(valor)

        if len(valor) == 0:
            raise ValueError("La lista de textos no puede estar vacia.")
        if len(valor) > MAX_DOCUMENTOS:
            raise ValueError(
                f"El lote excede el limite de {MAX_DOCUMENTOS} documentos."
            )
        for i, elemento in enumerate(valor):
            _validar_texto(elemento, f"texto en la posicion {i}")
        return valor

    def como_lista(self) -> List[str]:
        """Normaliza la entrada a lista para unificar el procesamiento interno."""
        return [self.text] if isinstance(self.text, str) else list(self.text)

    model_config = {
        "json_schema_extra": {
            "example": {"text": "Juan Pérez viajó a Bogotá con Ecopetrol."}
        }
    }


class PeticionTextoUnico(BaseModel):
    """
    Peticion para /visualize/dep, que procesa un unico documento por solicitud.

    El campo se declara como StrictStr y no admite lista: el uso de batch en
    este endpoint esta explicitamente fuera del contrato.
    """

    text: StrictStr

    @field_validator("text")
    @classmethod
    def validar(cls, valor: str) -> str:
        return _validar_texto(valor)

    model_config = {
        "json_schema_extra": {"example": {"text": "El gato come pescado."}}
    }


class PeticionVectorizar(BaseModel):
    """Peticion de vectorizacion, que requiere al menos dos documentos."""

    documents: List[StrictStr]

    @field_validator("documents")
    @classmethod
    def validar(cls, valor: List[str]) -> List[str]:
        if len(valor) < MIN_DOCUMENTOS_VECTORIZAR:
            raise ValueError(
                f"Se requieren al menos {MIN_DOCUMENTOS_VECTORIZAR} documentos."
            )
        if len(valor) > MAX_DOCUMENTOS:
            raise ValueError(
                f"La coleccion excede el limite de {MAX_DOCUMENTOS} documentos."
            )
        for i, documento in enumerate(valor):
            _validar_texto(documento, f"documento en la posicion {i}")
        return valor

    model_config = {
        "json_schema_extra": {
            "example": {
                "documents": [
                    "El gato come pescado y el gato duerme.",
                    "Juan come en Bogotá.",
                ]
            }
        }
    }
