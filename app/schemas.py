"""
Esquemas de entrada y salida de la API.

Se usa Pydantic para validar las peticiones antes de que lleguen a la logica de
negocio. Esto evita que texto vacio o parametros invalidos alcancen el pipeline
de spaCy, y hace que FastAPI genere automaticamente la documentacion OpenAPI
con ejemplos.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

MAX_LONGITUD_TEXTO = 100_000
MAX_DOCUMENTOS = 500


class OpcionesPreprocesamiento(BaseModel):
    """Configuracion de las fases de limpieza y transformacion."""

    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_emojis: bool = True
    remove_numbers: bool = False
    remove_accents: bool = False
    lowercase: bool = True
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    lemmatize: bool = True
    allowed_pos: Optional[List[str]] = Field(
        default=None,
        description="Si se especifica, conserva unicamente estas categorias "
        "gramaticales. Ejemplo: ['NOUN', 'VERB', 'ADJ'].",
    )


class PeticionTexto(BaseModel):
    """Peticion con un unico texto a analizar."""

    text: str = Field(..., description="Texto a procesar.")
    model: Optional[str] = Field(
        default=None,
        description="Modelo de spaCy a utilizar. Por defecto es_core_news_sm.",
    )

    @field_validator("text")
    @classmethod
    def validar_texto(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("El campo 'text' no puede estar vacio.")
        if len(v) > MAX_LONGITUD_TEXTO:
            raise ValueError(
                f"El texto excede el limite de {MAX_LONGITUD_TEXTO} caracteres."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Juan Pérez viajó a Bogotá el 5 de marzo para reunirse "
                "con Ecopetrol. Más info en https://ejemplo.com"
            }
        }
    }


class PeticionPreprocesamiento(PeticionTexto):
    """Peticion de preprocesamiento con opciones configurables."""

    options: OpcionesPreprocesamiento = Field(
        default_factory=OpcionesPreprocesamiento
    )


class PeticionCodificacion(BaseModel):
    """Peticion de codificacion sobre una coleccion de documentos."""

    documents: List[str] = Field(
        ..., description="Coleccion de documentos a codificar."
    )
    method: str = Field(
        default="tfidf",
        description="Metodo de codificacion: one_hot, bow o tfidf.",
    )
    ngram_min: int = Field(default=1, ge=1, le=5)
    ngram_max: int = Field(default=1, ge=1, le=5)
    max_features: Optional[int] = Field(default=None, ge=1)
    preprocess: bool = Field(
        default=False,
        description="Si es true, aplica limpieza y lematizacion a cada "
        "documento antes de codificar.",
    )
    compare_all: bool = Field(
        default=False,
        description="Si es true, ignora 'method' y devuelve las tres "
        "codificaciones para comparacion.",
    )

    @field_validator("documents")
    @classmethod
    def validar_documentos(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Debe proporcionar al menos un documento.")
        if len(v) > MAX_DOCUMENTOS:
            raise ValueError(
                f"El numero de documentos excede el limite de {MAX_DOCUMENTOS}."
            )
        if all(not d.strip() for d in v):
            raise ValueError("Todos los documentos estan vacios.")
        return v

    @field_validator("method")
    @classmethod
    def validar_metodo(cls, v: str) -> str:
        validos = {"one_hot", "bow", "tfidf"}
        if v not in validos:
            raise ValueError(
                f"Metodo invalido: '{v}'. Valores validos: {', '.join(sorted(validos))}."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "documents": [
                    "gato gato gato comer pescado",
                    "juan comer bogota",
                    "caballo comer rapido",
                ],
                "method": "tfidf",
            }
        }
    }


class RespuestaSalud(BaseModel):
    """Estado del servicio."""

    status: str
    modelo: str
    modelo_cargado: bool
    version_spacy: str
    entorno: str


class RespuestaError(BaseModel):
    """Formato uniforme de error."""

    detail: str
    tipo: Optional[str] = None
