"""
Vectorizacion de documentos: One-Hot, Bag of Words y TF-IDF.

Implementa las reglas de la seccion 4 de la guia utilizando scikit-learn.

Correspondencia entre la guia y los parametros de scikit-learn
--------------------------------------------------------------

La guia fija idf(t) = ln((|D| + 1) / (n_t + 1)) + 1. Esa es exactamente la
formula que aplica TfidfVectorizer con smooth_idf=True, que es su valor por
defecto. La equivalencia se verifica de forma automatica en la prueba
test_vectorize_tfidf_formula.

Los parametros se fijan de forma explicita porque los valores por defecto de la
libreria NO cumplen el contrato:

  norm=None
      Por defecto TfidfVectorizer aplica normalizacion L2 a cada fila. La guia
      exige el producto tf x idf "sin normalizacion posterior", de modo que hay
      que desactivarla.

  analyzer=_analizador (division por espacios)
      El patron de tokenizacion por defecto descarta los terminos de un solo
      caracter y volveria a segmentar el texto. Como la entrada ya viene limpia
      y con los terminos separados por un unico espacio, dividir por espacios
      preserva exactamente los terminos resultantes de la limpieza, que son los
      que la guia manda usar para construir el vocabulario.

  binary=False en CountVectorizer
      Bag of Words debe contener la frecuencia absoluta, no la presencia.

El orden lexicografico ascendente del vocabulario lo garantiza scikit-learn:
get_feature_names_out devuelve los terminos ordenados.

One-Hot se calcula aparte. La guia lo define como un vector por OCURRENCIA
retenida, produciendo una matriz por documento; scikit-learn no ofrece esa
representacion, ya que CountVectorizer(binary=True) genera un unico vector de
presencia por documento.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from app.core import cleaning

DECIMALES = 4


def _analizador(texto: str) -> List[str]:
    """
    Divide un texto ya limpio en sus terminos.

    Se define como funcion con nombre en lugar de pasar str.split directamente
    para dejar documentado el motivo de la eleccion.
    """
    return texto.split()


def construir_bow(documentos_limpios: List[str]):
    """
    Ajusta CountVectorizer y devuelve la matriz de frecuencias y el vocabulario.

    La matriz tiene dimension N x |V|, con las filas en el orden de los
    documentos recibidos y las columnas en el orden lexicografico del
    vocabulario.
    """
    vectorizador = CountVectorizer(analyzer=_analizador, binary=False)
    matriz = vectorizador.fit_transform(documentos_limpios)
    vocabulario = vectorizador.get_feature_names_out().tolist()
    return matriz.toarray().tolist(), vocabulario


def construir_tfidf(documentos_limpios: List[str]) -> List[List[float]]:
    """
    Calcula la matriz TF-IDF conforme a la formula de la guia.

    smooth_idf=True produce idf(t) = ln((|D|+1)/(n_t+1)) + 1 y norm=None evita
    la normalizacion L2 que la libreria aplicaria por defecto. El resultado se
    redondea a cuatro cifras decimales.
    """
    vectorizador = TfidfVectorizer(
        analyzer=_analizador,
        norm=None,
        use_idf=True,
        smooth_idf=True,
    )
    matriz = vectorizador.fit_transform(documentos_limpios).toarray()
    return [[round(float(valor), DECIMALES) for valor in fila] for fila in matriz]


def construir_one_hot(
    documentos_limpios: List[str], vocabulario: List[str]
) -> List[List[List[int]]]:
    """
    Genera una matriz por documento con un vector binario por ocurrencia.

    Cada fila corresponde a una ocurrencia retenida del documento y contiene un
    unico valor 1 en la posicion que el termino ocupa en el vocabulario. Un
    documento con k ocurrencias produce una matriz de dimension k x |V|.

    Un documento cuya limpieza no deja ningun termino produce una matriz vacia,
    lo que preserva la correspondencia posicional con la entrada.
    """
    indice = {termino: i for i, termino in enumerate(vocabulario)}
    salida: List[List[List[int]]] = []

    for texto in documentos_limpios:
        matriz_documento: List[List[int]] = []
        for termino in cleaning.terminos(texto):
            posicion = indice.get(termino)
            if posicion is None:
                continue
            vector = [0] * len(vocabulario)
            vector[posicion] = 1
            matriz_documento.append(vector)
        salida.append(matriz_documento)

    return salida


def vectorizar(documentos: List[str]) -> Dict[str, Any]:
    """
    Ejecuta la vectorizacion completa sobre una coleccion de documentos.

    Los documentos se limpian primero, ya que el vocabulario se construye con
    los terminos resultantes de la limpieza.

    Si tras la limpieza no queda ningun termino en toda la coleccion, se
    devuelven estructuras vacias coherentes en lugar de propagar la excepcion de
    vocabulario vacio que lanzaria scikit-learn.
    """
    limpios = cleaning.limpiar_lote(documentos)

    if not any(texto.strip() for texto in limpios):
        return {
            "vocabulary": [],
            "one_hot": [[] for _ in documentos],
            "bag_of_words": [[] for _ in documentos],
            "tf_idf": [[] for _ in documentos],
        }

    bag_of_words, vocabulario = construir_bow(limpios)

    return {
        "vocabulary": vocabulario,
        "one_hot": construir_one_hot(limpios, vocabulario),
        "bag_of_words": bag_of_words,
        "tf_idf": construir_tfidf(limpios),
    }
