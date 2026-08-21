"""
Punto de entrada para AWS Lambda.

Mangum adapta la interfaz ASGI de FastAPI al formato de eventos de Lambda. La
logica funcional es la misma que se sirve por Uvicorn en EC2: solo cambia el
transporte.

Handler configurado en Lambda: handler.handler
"""

from mangum import Mangum

from app.api import app

# lifespan="off" es necesario en Lambda: el ciclo de vida ASGI no encaja con el
# modelo de invocacion. El modelo de spaCy se carga de forma perezosa en la
# primera peticion y permanece en memoria mientras el contenedor siga vivo.
handler = Mangum(app, lifespan="off")
