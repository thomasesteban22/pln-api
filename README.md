# Microservicio de Procesamiento de Lenguaje Natural

**Laboratorio I — 2026 S02**
Universidad Sergio Arboleda
Escuela de Ciencias Exactas e Ingeniería
Programa de Ciencias de la Computación e Inteligencia Artificial

---

## 1. Descripción

Microservicio REST que expone cinco capacidades de procesamiento de lenguaje
natural sobre texto en español, implementado con FastAPI y spaCy
(`es_core_news_sm`).

La misma solución se despliega en dos arquitecturas dentro de AWS Academy:

1. **EC2** — despliegue persistente servido por Uvicorn bajo un servicio de
   systemd, de modo que arranca automáticamente tras un reinicio y se reinicia
   si el proceso muere.
2. **AWS Lambda** — despliegue serverless mediante imagen de contenedor,
   expuesto públicamente por Lambda Function URL.

La lógica funcional (`app/`) es común a ambos despliegues. Los archivos
específicos de cada arquitectura viven en `deploy/ec2/` y `deploy/lambda/`, de
modo que las diferencias de infraestructura no alteran el contrato observable
del servicio.

---

## 2. URLs de despliegue

| Arquitectura | URL |
|---|---|
| EC2 | http://100.55.30.215:8000 |
| Lambda | https://l76kxil2v3ssz6zvtdcar4rpl40symih.lambda-url.us-east-1.on.aws |

Comprobación rápida de disponibilidad:

```bash
curl http://184.73.114.15:8000/health
curl https://l76kxil2v3ssz6zvtdcar4rpl40symih.lambda-url.us-east-1.on.aws/health
```

Ambas responden `{"status":"ok","model":"es_core_news_sm","model_loaded":true,...}`,
diferenciándose únicamente en el campo `environment` (`ec2` o `lambda`).

---

## 3. Contrato

| Capacidad | Método y ruta | Entrada | Salida | Tipo |
|---|---|---|---|---|
| Limpieza | `POST /api/v1/clean` | `text`: string o lista | `cleaned_text`: lista | `application/json` |
| POS | `POST /api/v1/pos` | `text`: string o lista | `results[].tokens[]` con `text`, `pos`, `lemma` | `application/json` |
| NER | `POST /api/v1/ner` | `text`: string o lista | `results[].entities[]` con `text`, `label`, `start`, `end` | `application/json` |
| Dependencias | `POST /api/v1/visualize/dep` | `text`: un único string | HTML con SVG de displaCy | `text/html` |
| Vectorización | `POST /api/v1/vectorize` | `documents`: ≥2 strings | `vocabulary`, `one_hot`, `bag_of_words`, `tf_idf` | `application/json` |

Adicionalmente, `GET /health` reporta el estado del servicio y `GET /docs`
expone la documentación interactiva. Ambos son auxiliares y no forman parte del
contrato evaluado.

### Ejemplos

```bash
curl -X POST http://localhost:8000/api/v1/clean \
  -H "Content-Type: application/json" \
  -d '{"text": "El gato come pescado."}'
# {"cleaned_text": ["gato come pescado"]}

curl -X POST http://localhost:8000/api/v1/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Juan Pérez viajó a Bogotá."}'
# {"results": [{"entities": [
#   {"text": "Juan Pérez", "label": "PER", "start": 0, "end": 10},
#   {"text": "Bogotá", "label": "LOC", "start": 19, "end": 25}]}]}

curl -X POST http://localhost:8000/api/v1/vectorize \
  -H "Content-Type: application/json" \
  -d '{"documents": ["El gato come pescado y el gato duerme.", "Juan come en Bogotá."]}'
```

---

## 4. Reglas de vectorización

Implementadas literalmente según la sección 4 de la guía:

- **Vocabulario**: construido con los términos resultantes de la limpieza, en
  orden lexicográfico ascendente. Determina el orden de las columnas de las
  tres representaciones.
- **Bag of Words**: matriz `N × |V|` con la frecuencia absoluta de cada término.
- **One-Hot**: lista de `N` matrices. Cada fila representa **una ocurrencia
  retenida** del documento, con un único valor 1 en la posición del término.
  Un documento con `k` ocurrencias produce una matriz `k × |V|`.
- **TF-IDF**: `tf(t,d) × idf(t)` con `idf(t) = ln((|D|+1)/(n_t+1)) + 1`, **sin
  normalización posterior**, redondeado a cuatro decimales.

### Correspondencia con scikit-learn

Bag of Words y TF-IDF se calculan con `CountVectorizer` y `TfidfVectorizer`. La
fórmula que exige la guía es exactamente la que aplica scikit-learn con
`smooth_idf=True`, pero los valores **por defecto no cumplen el contrato**, de
modo que los parámetros se fijan explícitamente:

| Parámetro | Valor | Motivo |
|---|---|---|
| `smooth_idf` | `True` | Produce `idf(t) = ln((|D|+1)/(n_t+1)) + 1`, la fórmula de la guía |
| `norm` | `None` | Por defecto aplica normalización L2; la guía exige `tf × idf` sin normalizar |
| `analyzer` | división por espacios | El patrón por defecto descarta términos de un carácter y volvería a segmentar el texto ya limpio |
| `binary` | `False` | Bag of Words debe contener frecuencia absoluta, no presencia |

La prueba `test_vectorize_idf_coincide_con_formula_de_la_guia` reconstruye el
IDF término a término a partir del texto limpio y lo compara con el cociente
entre `tf_idf` y la frecuencia absoluta, verificando de forma automática que la
configuración elegida reproduce la fórmula exigida.

**One-Hot se calcula aparte.** La guía lo define como un vector por *ocurrencia*
retenida, produciendo una matriz por documento. scikit-learn no ofrece esa
representación: `CountVectorizer(binary=True)` genera un único vector de
presencia por documento, que es una representación distinta.

---

## 5. Decisiones de diseño

**La limpieza no se aplica antes del análisis lingüístico.** POS, NER y la
visualización de dependencias operan sobre el texto original. Los índices de las
entidades deben referirse al texto original (`start` inclusivo, `end`
exclusivo), y limpiar antes destruiría esa correspondencia. Además, el
reconocedor de entidades usa la capitalización como señal principal y el
analizador de dependencias usa la puntuación para delimitar cláusulas.

**La puntuación se sustituye por espacios antes de tokenizar.** El tokenizador
de spaCy separa `casa,perro` en tres tokens pero deja `gato;pez` como uno solo,
lo que produciría el término `gato;pez` y violaría la regla de que los signos de
puntuación actúan como separadores y no deben concatenar términos. La
sustitución previa garantiza el comportamiento exigido para cualquier signo. Se
seleccionan los caracteres de categoría Unicode `P`, de modo que letras
acentuadas, `ñ` y dígitos no se ven afectados.

**La validación ocurre antes del procesamiento.** Los esquemas Pydantic usan
`StrictStr` para rechazar coerción de tipos, y validan el lote completo antes de
que llegue al pipeline de spaCy. Así, un lote con un solo elemento inválido se
rechaza entero y no genera resultados parciales.

**El modelo se carga como singleton perezoso.** En Lambda el contenedor se
reutiliza entre invocaciones, de modo que el costo de carga se paga solo en el
arranque en frío.

---

## 6. Despliegue

### EC2

La instancia se aprovisiona de forma automática en el primer arranque mediante
`deploy/ec2/user-data.sh`, que se pega en el campo *Datos de usuario* del
asistente de lanzamiento. El script instala las dependencias del sistema, clona
el repositorio, crea el entorno virtual y registra el microservicio como
servicio de systemd.

Configuración de la instancia:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| AMI | Ubuntu Server **24.04 LTS** | Trae Python 3.12; ver la advertencia siguiente |
| Tipo | `t3.small` o superior | Con 1 GB de RAM la instalación de spaCy y scikit-learn se queda sin memoria |
| Almacenamiento | 30 GB | Las capas de la imagen de contenedor no caben en los 8 GB por defecto |
| Perfil de IAM | `LabInstanceProfile` | Necesario para AWS CLI y Session Manager |
| Puertos | 22 y 8000 desde `0.0.0.0/0` | La guía exige acceso desde una red externa |

**Advertencia sobre la versión de Ubuntu.** `spacy==3.8.3` declara
`Requires-Python <3.13`. Ubuntu 26.04 trae Python 3.14, de modo que pip descarta
todas las versiones compatibles y la instalación falla con *No matching
distribution found*. Fijar la versión de la librería no basta: la versión del
intérprete del sistema base es la que decide. El despliegue en Lambda no
presenta este problema porque la imagen fija `python:3.12` de forma explícita.

Para operar el servicio manualmente:

```bash
sudo systemctl status pln-api
sudo systemctl restart pln-api
sudo journalctl -u pln-api -n 50
```

Alternativa sin systemd, útil para desarrollo:

```bash
./deploy/ec2/run.sh
```

### Lambda

```bash
./deploy/lambda/deploy.sh
```

Construye la imagen, la publica en ECR, crea o actualiza la función y expone la
Function URL.

Detalles que suelen fallar y que el script ya contempla:

- **`--provenance=false`** en el build: sin esa bandera Docker genera un
  manifiesto multiplataforma que Lambda rechaza.
- **`--platform linux/amd64`** explícito: obligatorio si se construye desde una
  máquina ARM.
- **Los dos permisos de la Function URL.** Desde octubre de 2025 AWS exige
  `lambda:InvokeFunctionUrl` **y** `lambda:InvokeFunction` en la política basada
  en recursos. Con solo el primero, la URL devuelve **403 Forbidden** aunque el
  tipo de autenticación sea `NONE`.
- **Memoria de 2048 MB**: la CPU asignada es proporcional a la memoria; por
  debajo de 1536 MB la carga de spaCy se vuelve lenta y arriesga el timeout.
- **`/tmp` es el único punto escribible** en Lambda; el Dockerfile redirige allí
  los directorios de caché.
- **`click` debe declararse explícitamente.** `spacy==3.8.3` ejecuta
  `from click import NoSuchOption` pero no declara `click` en sus metadatos:
  históricamente lo recibía de forma transitiva a través de `typer`. Las
  versiones actuales de `typer` (0.27) y `typer-slim` (0.24) dejaron de
  declararlo, de modo que pip no lo instala y la imagen falla al importar spaCy.
  El despliegue sobre EC2 no lo manifiesta porque allí `uvicorn` lo arrastra.
- **`buildx` en Ubuntu.** El paquete `docker.io` de Ubuntu usa el constructor
  antiguo, que no reconoce `--provenance`. Se resuelve con
  `sudo apt-get install -y docker-buildx`.

---

## 7. Pruebas

### Resultados obtenidos

| Suite | Contra qué se ejecutó | Resultado |
|---|---|---|
| `test_contrato.py` | Aplicación en memoria | **49 passed** |
| `test_contrato.py` | EC2 (`http://184.73.114.15:8000`) | **49 passed** |
| `test_contrato.py` | Lambda (Function URL) | **49 passed** |
| `test_paridad.py` | Ambos despliegues simultáneamente | **17 passed** |

### Cómo reproducirlos

```bash
cd /home/ubuntu/pln-api
source .venv/bin/activate

# 1. Contrato sobre la aplicación en memoria
pytest tests/test_contrato.py -q

# 2. Contrato contra el despliegue en EC2
API_URL=http://184.73.114.15:8000 pytest tests/test_contrato.py -q

# 3. Contrato contra el despliegue en Lambda
API_URL=https://l76kxil2v3ssz6zvtdcar4rpl40symih.lambda-url.us-east-1.on.aws \
  pytest tests/test_contrato.py -q

# 4. Paridad entre ambos despliegues
EC2_URL=http://184.73.114.15:8000 \
LAMBDA_URL=https://l76kxil2v3ssz6zvtdcar4rpl40symih.lambda-url.us-east-1.on.aws \
  pytest tests/test_paridad.py -v
```

### Cobertura

`test_contrato.py` contiene 49 pruebas de caja negra. El mismo archivo sirve
para los tres escenarios: si la variable `API_URL` está definida se usan
peticiones HTTP reales; si no, se ejercita la aplicación en memoria. Cubre:

| Área | Qué se verifica |
|---|---|
| Limpieza | Minúsculas, eliminación de stop words y puntuación, conservación de tildes, `ñ` y dígitos, normalización de espacios, la puntuación como separador, lotes en orden |
| POS | Estructura `text`/`pos`/`lemma`, orden de tokens, lematización, correspondencia en lotes |
| NER | Estructura de entidades, offsets exactos sobre el texto original, texto sin entidades, correspondencia en lotes |
| Dependencias | Respuesta `text/html` con SVG, rechazo de lotes |
| Vectorización | Vocabulario lexicográfico, dimensiones `N × |V|`, frecuencia absoluta en BoW, One-Hot como lista de matrices con una fila por ocurrencia, fórmula del IDF término a término, ausencia de normalización, redondeo a 4 decimales, orden de filas, vocabulario vacío |
| Entradas inválidas | Las 13 formas enumeradas en la sección 9 de la guía, más el rechazo completo de un lote con un elemento inválido |
| Atributos de calidad | Consistencia entre solicitudes, 25 documentos para limpieza/POS/NER, 10 para vectorización, 5 solicitudes concurrentes e independientes, límite de 10 segundos por solicitud |

`test_paridad.py` envía las mismas peticiones a ambos despliegues y compara las
respuestas completas, incluidas las entradas inválidas. Incluye una
comprobación de que las dos URLs corresponden efectivamente a arquitecturas
distintas (`environment` igual a `ec2` y a `lambda`), para descartar el falso
positivo de apuntar ambas variables al mismo servicio.

### Demostración manual

```bash
API=http://184.73.114.15:8000

curl -X POST $API/api/v1/clean -H "Content-Type: application/json" \
  -d '{"text": "El gato come pescado."}'

curl -X POST $API/api/v1/clean -H "Content-Type: application/json" \
  -d '{"text": ["Compré 25 kilos de ñame.", "casa,perro. gato;pez"]}'

curl -X POST $API/api/v1/pos -H "Content-Type: application/json" \
  -d '{"text": "Los gatos corrían rápidamente por los tejados"}'

curl -X POST $API/api/v1/ner -H "Content-Type: application/json" \
  -d '{"text": "Juan Pérez viajó a Bogotá con Ecopetrol."}'

curl -X POST $API/api/v1/visualize/dep -H "Content-Type: application/json" \
  -d '{"text": "El gato come pescado."}'

curl -X POST $API/api/v1/vectorize -H "Content-Type: application/json" \
  -d '{"documents": ["El gato come pescado y el gato duerme.", "Juan come en Bogotá."]}'

# Entrada inválida: debe responder 4xx sin resultados parciales
curl -i -X POST $API/api/v1/clean -H "Content-Type: application/json" \
  -d '{"text": ["texto valido", "   "]}'
```

---

## 8. Estructura

```
.
├── app/                        Lógica funcional común a ambos despliegues
│   ├── core/
│   │   ├── pipeline.py         Carga singleton del modelo de spaCy
│   │   ├── cleaning.py         Limpieza y normalización
│   │   ├── linguistics.py      POS, NER y displaCy
│   │   └── vectorize.py        One-Hot, BoW y TF-IDF
│   ├── schemas.py              Validación de entrada
│   └── api.py                  Aplicación FastAPI y rutas del contrato
├── deploy/
│   ├── ec2/run.sh              Despliegue persistente
│   └── lambda/
│       ├── Dockerfile          Imagen de contenedor
│       ├── handler.py          Adaptador Mangum
│       └── deploy.sh           Publicación en ECR y Function URL
├── tests/
│   ├── test_contrato.py        47 pruebas de caja negra
│   └── test_paridad.py         Comparación entre despliegues
└── requirements.txt
```

---

## 9. Declaración de uso de inteligencia artificial

Conforme a la sección 6 de la guía.

**Herramienta utilizada:** Claude (Anthropic).

**Propósito:** apoyo en el diseño de la arquitectura del microservicio,
redacción del código de los módulos de limpieza, análisis lingüístico y
vectorización, elaboración de la suite de pruebas, configuración del despliegue
en contenedor y diagnóstico de errores de infraestructura en AWS.

**Forma de verificación de los resultados:**

1. **Verificación por pruebas automatizadas.** Toda la funcionalidad está
   cubierta por las 47 pruebas de `test_contrato.py`, ejecutadas tanto sobre la
   aplicación en memoria como contra los dos despliegues reales por HTTP.

2. **Verificación manual de las fórmulas.** Los valores de TF-IDF se
   comprobaron contra el cálculo hecho a mano de `idf(t) = ln((|D|+1)/(n_t+1)) + 1`.
   Por ejemplo, para un término que aparece dos veces en uno de tres documentos:
   `2 × (ln(4/2) + 1) = 3.3863`, valor que coincide con el devuelto por el
   servicio y que la prueba `test_vectorize_tfidf_formula` verifica de forma
   automática.

3. **Verificación del comportamiento de spaCy sobre casos concretos.** Se
   comprobó empíricamente que el tokenizador no separa `gato;pez`, hallazgo que
   motivó la normalización previa de puntuación y que quedó registrado en la
   prueba `test_clean_puntuacion_no_concatena_terminos`.

4. **Verificación de la paridad entre despliegues.** `test_paridad.py` compara
   las respuestas de EC2 y Lambda ante las mismas solicitudes, incluidas las
   entradas inválidas.

5. **Corrección de recomendaciones incorrectas.** Durante el desarrollo se
   descartaron por comprobación empírica varias hipótesis erróneas sugeridas
   inicialmente por la herramienta. Los tres casos más relevantes:

   - Un error **403** en la Function URL se atribuyó primero a una restricción
     de la cuenta de AWS Academy. La consulta a la documentación oficial reveló
     que la causa real era la ausencia del permiso `lambda:InvokeFunction`,
     exigido desde octubre de 2025 además de `lambda:InvokeFunctionUrl`.
   - Un `ModuleNotFoundError: No module named 'click'` en la construcción de la
     imagen se atribuyó sucesivamente al comportamiento de `pip --target` y a
     las rutas de `/var/runtime`. Ambas hipótesis se descartaron ejecutando la
     resolución de dependencias con `pip install --dry-run --report`, que
     mostró los 41 paquetes resueltos sin `click` entre ellos, y consultando en
     PyPI los metadatos de `spacy`, `typer` y `typer-slim`.
   - Una propuesta de desactivar el apagado automático de Cloud9 escribiendo
     `SHUTDOWN_TIMEOUT=0` se descartó tras leer el script
     `/opt/c9/stop-if-inactive.sh`: el valor `0` habría ejecutado
     `shutdown -h 0`, apagando la instancia de inmediato.

   El criterio adoptado tras estos casos fue no aceptar diagnósticos por
   plausibilidad, sino obtener la evidencia que los confirma o los refuta antes
   de aplicar cualquier corrección.

No se compartieron con la herramienta contraseñas, claves de acceso, tokens,
credenciales ni información sensible de AWS Academy. El repositorio no contiene
credenciales.
