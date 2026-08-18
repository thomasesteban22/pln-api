"""
Codificacion de texto: one-hot, Bag-of-Words y TF-IDF.

Convierte un conjunto de documentos en una representacion numerica apta para un
algoritmo de aprendizaje automatico.

Nota terminologica relevante: lo que aqui se denomina 'one_hot' es la variante
a nivel de documento, es decir un vector binario de presencia/ausencia sobre el
vocabulario (formalmente multi-hot). El one-hot estricto asigna un vector de
tamano |V| a cada token individual, con un unico 1. Se expone tambien esa
variante bajo 'one_hot_por_token' para que la distincion quede explicita.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

METODOS_VALIDOS = ("one_hot", "bow", "tfidf")


def _build_vectorizer(
    metodo: str,
    ngram_range: Tuple[int, int],
    max_features: Optional[int],
    vocabulary: Optional[List[str]],
):
    """
    Construye el vectorizador de scikit-learn correspondiente al metodo pedido.

    Para TF-IDF se usa smooth_idf=True (comportamiento por defecto de
    scikit-learn). Conviene recordar que scikit-learn siempre suma 1 al IDF
    final, de forma que un termino presente en todos los documentos recibe peso
    1.0 y no 0.0 como en la formula clasica log(N/df).
    """
    comun = {
        "ngram_range": ngram_range,
        "max_features": max_features,
        "vocabulary": vocabulary,
        "lowercase": True,
        "token_pattern": r"(?u)\b\w+\b",  # conserva tokens de un solo caracter
    }

    if metodo == "one_hot":
        return CountVectorizer(binary=True, **comun)
    if metodo == "bow":
        return CountVectorizer(binary=False, **comun)
    if metodo == "tfidf":
        return TfidfVectorizer(norm="l2", use_idf=True, smooth_idf=True, **comun)

    raise ValueError(
        f"Metodo de codificacion no soportado: '{metodo}'. "
        f"Valores validos: {', '.join(METODOS_VALIDOS)}."
    )


def encode(
    documents: List[str],
    metodo: str = "tfidf",
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: Optional[int] = None,
    vocabulary: Optional[List[str]] = None,
    incluir_matriz: bool = True,
) -> Dict[str, Any]:
    """
    Codifica una lista de documentos con el metodo indicado.

    Devuelve el vocabulario resultante, la matriz documento-termino y, en el
    caso de TF-IDF, los valores IDF por termino, que permiten interpretar el
    peso asignado a cada palabra.
    """
    if not documents:
        raise ValueError("La lista de documentos no puede estar vacia.")
    if any(not isinstance(d, str) for d in documents):
        raise ValueError("Todos los documentos deben ser cadenas de texto.")
    if all(not d.strip() for d in documents):
        raise ValueError("Todos los documentos estan vacios tras el recorte.")

    vectorizer = _build_vectorizer(metodo, ngram_range, max_features, vocabulary)
    matriz = vectorizer.fit_transform(documents)
    terminos = vectorizer.get_feature_names_out().tolist()

    resultado: Dict[str, Any] = {
        "metodo": metodo,
        "num_documentos": len(documents),
        "vocabulario": terminos,
        "tamano_vocabulario": len(terminos),
        "dimensiones_matriz": list(matriz.shape),
        "ngram_range": list(ngram_range),
        "densidad": round(float(matriz.nnz) / (matriz.shape[0] * matriz.shape[1]), 4)
        if matriz.shape[1] > 0
        else 0.0,
    }

    if incluir_matriz:
        denso = matriz.toarray()
        resultado["matriz"] = [
            [round(float(v), 6) for v in fila] for fila in denso
        ]
        resultado["documentos"] = [
            {
                "indice": i,
                "texto": documents[i],
                "vector": dict(
                    (terminos[j], round(float(denso[i][j]), 6))
                    for j in range(len(terminos))
                    if denso[i][j] != 0
                ),
            }
            for i in range(len(documents))
        ]

    if metodo == "tfidf":
        resultado["idf"] = {
            termino: round(float(peso), 6)
            for termino, peso in zip(terminos, vectorizer.idf_)
        }

    return resultado


def one_hot_por_token(
    documents: List[str], vocabulary: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    One-hot en sentido estricto: un vector por token, no por documento.

    Cada token se representa como un vector de longitud |V| con un unico 1 en la
    posicion correspondiente a su indice en el vocabulario. Se incluye para
    contrastar con la variante de presencia/ausencia a nivel de documento y
    evidenciar el costo en dimensionalidad de esta representacion.
    """
    if vocabulary is None:
        vistos: List[str] = []
        for doc in documents:
            for token in doc.lower().split():
                if token not in vistos:
                    vistos.append(token)
        vocabulary = vistos

    indice = {termino: i for i, termino in enumerate(vocabulary)}
    tamano = len(vocabulary)

    codificados = []
    for i, doc in enumerate(documents):
        vectores = []
        for token in doc.lower().split():
            vector = [0] * tamano
            if token in indice:
                vector[indice[token]] = 1
            vectores.append({"token": token, "vector": vector})
        codificados.append({"indice_documento": i, "tokens": vectores})

    return {
        "metodo": "one_hot_por_token",
        "vocabulario": list(vocabulary),
        "tamano_vocabulario": tamano,
        "documentos": codificados,
    }


def comparar_metodos(
    documents: List[str], ngram_range: Tuple[int, int] = (1, 1)
) -> Dict[str, Any]:
    """
    Ejecuta las tres codificaciones sobre el mismo corpus y las devuelve juntas.

    Pensado para el analisis comparativo one-hot vs BoW vs TF-IDF: al compartir
    corpus y vocabulario, las matrices son directamente contrastables termino a
    termino.
    """
    return {
        "corpus": documents,
        "resultados": {
            metodo: encode(documents, metodo=metodo, ngram_range=ngram_range)
            for metodo in METODOS_VALIDOS
        },
    }
