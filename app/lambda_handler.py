"""
Punto de entrada para AWS Lambda.

Mangum adapta la interfaz ASGI de FastAPI al formato de eventos de Lambda,
permitiendo que la misma aplicacion se sirva tanto por Uvicorn en Cloud9/EC2
como por Lambda detras de un Function URL o de API Gateway.

El nombre completo del handler que debe configurarse en Lambda es:
    app.lambda_handler.handler
"""

from mangum import Mangum

from app.main import app

# lifespan="off" es necesario en Lambda: el ciclo de vida ASGI no encaja con el
# modelo de invocacion de la funcion. El modelo de spaCy se carga de forma
# perezosa en la primera peticion y permanece en memoria mientras el contenedor
# de ejecucion siga vivo.
handler = Mangum(app, lifespan="off")
