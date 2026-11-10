import glob
import pathlib
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager


# Variante global para mantener el modelo cargado en la memoria
model = None 

def find_latest_mlflow_model() -> str:
    """Busca automáticamente el último modelo entrenado guardado dentro de la carpeta mlruns/"""
    # Buscamos carpetas de artefactos de modelo en la estructura de MLflow
    model_paths = glob.glob("mlruns/*/*/artifacts/model")

    if not model_paths:
        raise FileNotFoundError(
            "No se encontró ningún modelo registrado en la carpeta 'mlruns/'. "
            "Asegúrate de haber ejecutado 'python src/models/train_mlflow.py' previamente."
        )

    # Ordenamos por fecha de modificacion para tomar el mas reciente
    model_paths.sort(key=lambda x: pathlib.Path(x).stat().st_mtime, reverse=True)
    return model_paths[0]

# 2. Definimos el esquema del cuerpo JSON que la api espera recibir (Validacion pydantic) 
class MarketFeatures (BaseModel):
    Return: float
    SMA_5: float
    SMA_20: float
    Volatility_5: float

# 3: Evento de inicio del servidor: Carga el modelo solo una vez
#@app.lifespan("start_up")
#def load_trained_model():
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Se ejecuta automátiamente al arrancar la API.

    Carga el modelo de MLflow a la RAM.
    """
    global model
    try: 
        latest_model_path = find_latest_mlflow_model()
        print(f" Cargando modelo servible desde: {latest_model_path}")

        # Cargamos el modelo físico scikit-learn usando la librería de MLflow
        model = mlflow.sklearn.load_model(latest_model_path)
        print(" Modelo cargado en memoria exitosamente. La API está lista.")

    except Exception as e:
        print(f"Advertencia no se pudo cargar el modelo fisico: ({e})")
        print(f" Se activara el modo de respaldo (Fallback) para atender peticiones")

    yield

app = FastAPI(
    title="S&P 500 Market Prediction API",
    description="API REST para predecir la direccion del indice S&P 500 usando MLOps",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def health_check():
    """ Ruta raíz para verificar que el servidor esté vivo. """
    return {
        "status": "online",
        "model_loaded": model is not None,
        "message": "Servicio de Predicción MLOps activo."
    }

@app.post("/predict")
def predict_market(data: MarketFeatures):
    """Ruta principal: Recibe las características del mercado en JSON y devuelve la prediccion """

    # Preparamos los datos de entrada en el formato DataFrame que espera el modelo
    input_data = pd.DataFrame([{
        "Return": data.Return,
        "SMA_5": data.SMA_5,
        "SMA_20": data.SMA_20,
        "Volatility_5": data.Volatility_5
    }])

    # Si el modelo cargó correctamente desde MLflow, usamos sus predicciones reales
    if model is not None:
        prediction = int(model.predict(input_data)[0])
        # Calculamos la probabilidad de que la clase 1 (que suba)
        probabilities = model.predict_proba(input_data)[0]
        prob_up = float(probabilities[1])

    else:
        # Modo de respaldo si no hay modelo fisico guardado
        prediction = 1 if data.Return > 0 else 0
        prob_up = 0.50

    return {
        "prediction": prediction,
        "label": "SUBIRA (1)" if prediction == 1 else "BAJARA/SE MANTIENE (0)",
        "probability_up": round(prob_up, 4),
        "input_received": data.model_dump()
    }