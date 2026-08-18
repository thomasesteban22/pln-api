#!/usr/bin/env python3
"""
Visualizacion comparativa de las codificaciones de texto.

Consulta el endpoint /encoding con compare_all activado y presenta las tres
representaciones (one-hot, Bag-of-Words y TF-IDF) como matrices documento-termino
alineadas, junto con los pesos IDF y un resumen cuantitativo.

Uso:
    python mostrar_codificacion.py
    python mostrar_codificacion.py "primer documento" "segundo documento"
    python mostrar_codificacion.py --archivo corpus.txt
    python mostrar_codificacion.py --preprocesar "Los gatos corrían" "El gato corre"
    python mostrar_codificacion.py --url http://localhost:8080 --ngramas 1 2

Con --archivo se lee un documento por linea.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

CORPUS_POR_DEFECTO = [
    "gato gato gato comer pescado",
    "juan comer bogota",
    "caballo comer rapido",
]

NOMBRES = {
    "one_hot": "ONE-HOT (presencia / ausencia)",
    "bow": "BAG-OF-WORDS (frecuencia absoluta)",
    "tfidf": "TF-IDF (frecuencia ponderada, norma L2)",
}


def separador(titulo: str, ancho: int = 78) -> None:
    """Imprime un encabezado de seccion delimitado."""
    print()
    print("=" * ancho)
    print(titulo)
    print("=" * ancho)


def formatear_valor(valor: float) -> str:
    """
    Da formato al valor segun su naturaleza.

    Los enteros se muestran sin decimales para que las matrices binarias y de
    conteo se lean con claridad; los valores fraccionarios se muestran con tres
    decimales. El cero se representa con un punto para que el patron de
    dispersion de la matriz resulte visible de un vistazo.
    """
    if valor == 0:
        return "."
    if float(valor).is_integer():
        return str(int(valor))
    return f"{valor:.3f}"


def imprimir_matriz(vocabulario: list, matriz: list, etiquetas: list) -> None:
    """
    Imprime la matriz documento-termino con columnas alineadas.

    El ancho de cada columna se calcula a partir del contenido real, de modo que
    la tabla se mantenga legible con vocabularios de longitudes dispares.
    """
    anchos = []
    for j, termino in enumerate(vocabulario):
        ancho_valores = max(
            (len(formatear_valor(fila[j])) for fila in matriz), default=1
        )
        anchos.append(max(len(termino), ancho_valores))

    ancho_etiqueta = max(len(e) for e in etiquetas)

    encabezado = " " * ancho_etiqueta + " |"
    for termino, ancho in zip(vocabulario, anchos):
        encabezado += f" {termino:>{ancho}}"
    print(encabezado)
    print("-" * len(encabezado))

    for etiqueta, fila in zip(etiquetas, matriz):
        linea = f"{etiqueta:<{ancho_etiqueta}} |"
        for valor, ancho in zip(fila, anchos):
            linea += f" {formatear_valor(valor):>{ancho}}"
        print(linea)


def imprimir_idf(idf: dict) -> None:
    """
    Imprime los pesos IDF ordenados de mayor a menor con una barra proporcional.

    El orden descendente hace inmediato el punto central del TF-IDF: los
    terminos mas discriminativos encabezan la lista y los que aparecen en todo
    el corpus quedan al final.
    """
    print()
    print("Pesos IDF (scikit-learn: log((1+N)/(1+df)) + 1):")
    print()

    maximo = max(idf.values()) if idf else 1
    ancho_termino = max(len(t) for t in idf)

    for termino, peso in sorted(idf.items(), key=lambda x: -x[1]):
        barra = "#" * int((peso / maximo) * 32)
        print(f"  {termino:<{ancho_termino}}  {peso:6.4f}  {barra}")


def imprimir_resumen(resultados: dict, corpus: list) -> None:
    """Compara las tres codificaciones sobre metricas estructurales comunes."""
    print()
    print(f"{'Metodo':<12} {'Dimensiones':<14} {'Densidad':<10} {'Valores unicos'}")
    print("-" * 60)

    for metodo in ("one_hot", "bow", "tfidf"):
        datos = resultados[metodo]
        valores = {v for fila in datos["matriz"] for v in fila}
        distintos = "2 (0 y 1)" if metodo == "one_hot" else str(len(valores))
        dimensiones = f"{datos['dimensiones_matriz'][0]} x {datos['dimensiones_matriz'][1]}"
        print(
            f"{metodo:<12} {dimensiones:<14} "
            f"{datos['densidad']:<10.4f} {distintos}"
        )

    print()
    print("Lectura:")
    print("  - One-hot y BoW comparten dimensiones y densidad; solo difieren en")
    print("    que BoW conserva la frecuencia y one-hot la descarta.")
    print("  - TF-IDF mantiene la misma estructura pero pondera cada termino")
    print("    segun su capacidad de discriminar entre documentos.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara one-hot, BoW y TF-IDF sobre un corpus."
    )
    parser.add_argument("documentos", nargs="*", help="Documentos a codificar.")
    parser.add_argument(
        "--url",
        default=os.environ.get("API_URL", "http://localhost:8080"),
        help="URL base de la API.",
    )
    parser.add_argument("--archivo", help="Archivo con un documento por linea.")
    parser.add_argument(
        "--preprocesar",
        action="store_true",
        help="Lematiza y limpia cada documento antes de codificar.",
    )
    parser.add_argument(
        "--ngramas",
        nargs=2,
        type=int,
        default=[1, 1],
        metavar=("MIN", "MAX"),
        help="Rango de n-gramas.",
    )
    args = parser.parse_args()

    if args.archivo:
        with open(args.archivo, encoding="utf-8") as fh:
            corpus = [linea.strip() for linea in fh if linea.strip()]
    elif args.documentos:
        corpus = args.documentos
    else:
        corpus = CORPUS_POR_DEFECTO

    peticion = {
        "documents": corpus,
        "compare_all": True,
        "preprocess": args.preprocesar,
        "ngram_min": args.ngramas[0],
        "ngram_max": args.ngramas[1],
    }

    try:
        respuesta = requests.post(
            f"{args.url.rstrip('/')}/encoding", json=peticion, timeout=60
        )
    except requests.exceptions.ConnectionError:
        print(f"No se pudo conectar con {args.url}.", file=sys.stderr)
        print("Verifique que el servidor este en ejecucion.", file=sys.stderr)
        return 1

    if respuesta.status_code != 200:
        print(f"La API devolvio {respuesta.status_code}:", file=sys.stderr)
        print(respuesta.text[:500], file=sys.stderr)
        return 1

    datos = respuesta.json()
    resultados = datos["resultados"]
    corpus_final = datos["corpus"]
    etiquetas = [f"d{i + 1}" for i in range(len(corpus_final))]

    separador("CORPUS")
    for etiqueta, documento in zip(etiquetas, corpus_final):
        print(f"  {etiqueta}: {documento}")
    if args.preprocesar:
        print()
        print("  (documentos lematizados antes de codificar)")

    vocabulario = resultados["bow"]["vocabulario"]
    print()
    print(f"Vocabulario ({len(vocabulario)} terminos): {', '.join(vocabulario)}")

    for metodo in ("one_hot", "bow", "tfidf"):
        separador(NOMBRES[metodo])
        imprimir_matriz(vocabulario, resultados[metodo]["matriz"], etiquetas)
        if metodo == "tfidf":
            imprimir_idf(resultados[metodo]["idf"])

    separador("COMPARACION")
    imprimir_resumen(resultados, corpus_final)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
