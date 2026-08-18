# API de Procesamiento de Lenguaje Natural

Universidad Sergio Arboleda
Escuela de Ciencias Exactas e Ingeniería
Programa de Ciencias de la Computación e Inteligencia Artificial
Asignatura: Procesamiento de Lenguaje Natural (PCIA5011)

---

## Descripción

Servicio REST que expone operaciones de procesamiento de lenguaje natural sobre
texto en español, implementado con FastAPI y spaCy. El mismo código de
aplicación se despliega en dos entornos de AWS:

1. **Cloud9 / EC2**, servido directamente por Uvicorn.
2. **AWS Lambda**, empaquetado como imagen de contenedor y adaptado mediante Mangum.

La equivalencia funcional entre ambos despliegues se verifica con una suite de
pruebas automatizadas que se ejecuta indistintamente contra cualquiera de las
dos URLs.

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Catálogo de endpoints |
| GET | `/health` | Estado del servicio y del modelo |
| GET | `/docs` | Documentación interactiva OpenAPI |
| POST | `/processed` | Limpieza y transformación de texto |
| POST | `/dependency` | Árbol de dependencias sintácticas |
| POST | `/ner` | Reconocimiento de entidades nombradas |
| POST | `/full` | Los tres análisis anteriores combinados |
| POST | `/encoding` | Codificación one-hot, BoW o TF-IDF |

### POST /processed

Ejecuta las dos fases del preprocesamiento por separado y documenta cada
transformación aplicada.

```json
{
  "text": "<p>Los gatos corrían por https://ejemplo.com 😀</p>",
  "options": { "lemmatize": true, "remove_stopwords": true }
}
```

La respuesta separa `limpieza` (con la traza de operaciones y el número de
caracteres afectados por cada una) y `transformacion` (tokens finales,
vocabulario y los tokens descartados con su motivo). Para el ejemplo anterior el
texto limpio es `los gatos corrían por` y los tokens procesados
`["gato", "correr"]`.

Opciones disponibles: `remove_html`, `remove_urls`, `remove_emails`,
`remove_emojis`, `remove_numbers`, `remove_accents`, `lowercase`,
`remove_stopwords`, `remove_punctuation`, `lemmatize`, `allowed_pos`.

### POST /dependency

```json
{ "text": "El gato come pescado." }
```

Devuelve, por cada token, su relación de dependencia, el núcleo del que depende,
sus hijos y la categoría gramatical, más la raíz de cada oración.

### POST /ner

```json
{ "text": "Juan Pérez viajó a Bogotá para reunirse con Ecopetrol." }
```

Produce el texto anotado en línea:

```
Juan Pérez [PER] viajó a Bogotá [LOC] para reunirse con Ecopetrol [ORG].
```

Además de la lista estructurada con offsets de carácter y el esquema BIO por
token, útil si posteriormente se entrena un modelo de secuencias.

### POST /full

Combina los tres análisis. Los análisis lingüísticos se aplican sobre el texto
**original**, no sobre el preprocesado: lematizar y eliminar stop words antes
del parser o del reconocedor de entidades degrada notablemente ambos resultados,
porque el parser depende de la puntuación para delimitar cláusulas y el NER usa
las mayúsculas como señal principal.

### POST /encoding

```json
{
  "documents": ["gato gato gato comer pescado", "juan comer bogota"],
  "method": "tfidf",
  "compare_all": false
}
```

Métodos: `one_hot` (presencia/ausencia), `bow` (frecuencia absoluta) y `tfidf`.
Con `compare_all: true` se devuelven las tres representaciones sobre el mismo
corpus. Con `preprocess: true` cada documento se lematiza antes de codificar.

**Nota terminológica.** Lo que aquí se denomina `one_hot` es la variante a nivel
de documento, es decir un vector binario de presencia/ausencia sobre el
vocabulario, que formalmente es *multi-hot*. El one-hot estricto asigna un
vector de tamaño |V| a cada token individual con un único 1. Esa variante está
disponible en `app/core/encoding.py::one_hot_por_token`.

**Sobre el IDF de scikit-learn.** La librería siempre suma 1 al IDF final, de
modo que la fórmula efectiva es `log(N/df) + 1` y no el clásico `log(N/df)`. Un
término presente en todos los documentos recibe por tanto peso 1.0 y no 0.0. En
el corpus de ejemplo, `comer` aparece en los tres documentos y obtiene el IDF
mínimo, mientras que `gato` obtiene el máximo.

---

## Ejecución local

```bash
./run_local.sh
```

O manualmente:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La documentación interactiva queda en `http://localhost:8000/docs`.

---

## Despliegue 1: Cloud9 / EC2

1. Crear un entorno de Cloud9 (Amazon Linux 2023, `t3.small` o superior; el
   modelo de spaCy más scikit-learn requieren más de 1 GB de RAM durante la
   instalación).

2. Clonar el proyecto y ejecutar:

   ```bash
   ./run_local.sh
   ```

3. Para acceso desde fuera de Cloud9, abrir el puerto en el Security Group de la
   instancia (entrada TCP 8000 desde `0.0.0.0/0`) y usar la IP pública.

4. Para vista previa dentro de Cloud9, usar el puerto 8080:

   ```bash
   PORT=8080 ./run_local.sh
   ```

**Advertencia sobre la instalación del modelo.** En Amazon Linux 2023 y en
Ubuntu 24 el comando `python -m spacy download es_core_news_sm` falla por la
restricción PEP 668 sobre entornos gestionados. Dentro de un entorno virtual
funciona sin problema; fuera de él hay que instalar el wheel directamente desde
la URL del release, que es exactamente lo que hace el `requirements.txt` de este
proyecto.

---

## Despliegue 2: Lambda con Docker

Lambda impone un límite de 250 MB al paquete ZIP descomprimido. spaCy con su
modelo y scikit-learn superan ese límite holgadamente, por lo que el despliegue
como **imagen de contenedor** (límite de 10 GB) no es una preferencia sino un
requisito técnico.

```bash
./deploy_lambda.sh
```

El script crea el repositorio de ECR, construye la imagen, la publica y
actualiza la función. Configuración recomendada:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Memoria | 2048 MB | La CPU asignada es proporcional a la memoria; por debajo de 1536 MB la carga de spaCy se vuelve muy lenta |
| Tiempo de espera | 60 s | Cubre el arranque en frío |
| Arquitectura | x86_64 | Debe coincidir con `--platform linux/amd64` |
| URL de la función | Auth `NONE` | Expone la API sin API Gateway |

### Detalles que suelen fallar

- **`--provenance=false` en el build.** Sin esa bandera, Docker genera un
  manifiesto multiplataforma que Lambda rechaza con un error poco descriptivo
  sobre el formato de la imagen.
- **`--platform linux/amd64` explícito.** Si se construye desde un equipo con
  arquitectura ARM (Apple Silicon), la imagen resultante no arranca en una
  función configurada como x86_64.
- **Sistema de archivos de solo lectura.** En Lambda únicamente `/tmp` es
  escribible. Por eso el `Dockerfile` fija `HF_HOME` y `XDG_CACHE_HOME` a `/tmp`.
- **`lifespan="off"` en Mangum.** El ciclo de vida ASGI no encaja con el modelo
  de invocación de Lambda. El modelo de spaCy se carga de forma perezosa en la
  primera petición y permanece en memoria mientras el contenedor de ejecución
  siga vivo, de modo que solo el arranque en frío paga ese costo.
- **Verificación en tiempo de construcción.** El `Dockerfile` carga el modelo
  durante el build. Si falta, la construcción falla ahí y no en producción.

---

## Pruebas automatizadas

Suite de 48 pruebas sobre la aplicación en memoria:

```bash
pytest tests/test_endpoints.py -v
```

Cobertura por área:

| Área | Qué se verifica |
|------|-----------------|
| General | Catálogo de endpoints, health check, cabecera de latencia |
| `/processed` | Eliminación de ruido, lematización, reducción de vocabulario, motivo de descarte por token, filtro por categoría gramatical, opciones configurables |
| `/dependency` | Unicidad de la raíz, estructura de cada token, segmentación de oraciones, validez de los índices de núcleo |
| `/ner` | Detección de entidades, marcas en el texto anotado, correspondencia exacta de offsets, esquema BIO, texto sin entidades |
| `/encoding` | Frecuencias de BoW, binariedad de one-hot, diferencia observable entre ambos, penalización del IDF, normalización L2, dimensiones, n-gramas, comparación de métodos |
| Transversal | Rechazo de texto vacío, tolerancia a Unicode y a textos largos |

### Pruebas contra el despliegue real

```bash
export API_URL=http://<ip-ec2>:8000
pytest tests/test_remoto.py -v

export API_URL=https://<id>.lambda-url.us-east-1.on.aws
pytest tests/test_remoto.py -v
```

Ejecutar la misma suite contra ambas URLs es lo que demuestra que los dos
despliegues son funcionalmente equivalentes.

---

## Estructura del proyecto

```
pln-api/
├── app/
│   ├── core/
│   │   ├── pipeline.py        Carga singleton del modelo de spaCy
│   │   ├── preprocessing.py   Limpieza y transformación
│   │   ├── linguistics.py     Dependencias y NER
│   │   └── encoding.py        One-hot, BoW y TF-IDF
│   ├── schemas.py             Validación de entrada y salida
│   ├── main.py                Aplicación FastAPI
│   └── lambda_handler.py      Adaptador Mangum
├── tests/
│   ├── test_endpoints.py      48 pruebas en memoria
│   └── test_remoto.py         Verificación de despliegue
├── Dockerfile                 Imagen de contenedor para Lambda
├── requirements.txt
├── run_local.sh
└── deploy_lambda.sh
```

---

## Decisiones de diseño

**Separación entre limpieza y transformación.** Las dos fases se implementan y
se reportan por separado porque responden a propósitos distintos: la primera
elimina ruido sobre la cadena cruda, la segunda normaliza el texto ya tokenizado
sin distorsionar el significado.

**El preprocesamiento no se aplica antes del análisis lingüístico.** Es la
decisión más relevante del diseño y la razón por la que `/full` no encadena los
endpoints. Un texto en minúsculas y sin puntuación reduce drásticamente la
calidad del parser y del NER.

**Trazabilidad de cada transformación.** Toda operación aplicada queda
registrada con su impacto, y cada token descartado indica su motivo. El
resultado es auditable y permite justificar las decisiones de preprocesamiento.

**Carga perezosa del modelo.** El singleton mediante `lru_cache` es lo que hace
viable el despliegue en Lambda: sin él, cada invocación pagaría el costo de
cargar spaCy.
