import math
import pathlib
import pandas as pd

"""Versión didáctica de train.py sin scikit-learn (Desde Cero)"""

""" 
    ¿Cómo funciona la Regresión Logística pura?

   1.- Asigna un peso (w) a cada variable (Retorno, Medias Móviles, Volatilidad) y un sesgo (b).
   2.- Multiplica las entradas por los pesos: $z = w_1 x_1 + w_2 x_2 + ... + b.
   3.- Aplica la Función Sigmoide para comprimir el resultado en una probabilidad entre $0.0$ y $1.0$.
   4.- Si la probabilidad es mayor a 0.5, predice $1$ (sube); si no, $0$ (baja).
   5.- Ajusta los pesos iterativamente mediante Descenso de Gradiente para reducir el error.
   
"""

# Funciones matematicas puras
def sigmoid(z: float) -> float:
    """ Funcion sigmoide: mapea cualquier numero real a un valor entre 0 y 1 """
    # Prevenimos desbordamiento numerico (overflow)
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def predict_probability(features: list[float], weights: list[float], bias: float) -> float:
    """ Calcula la probabilidad P(Y=1|X) mediante la combinacion lineal + sigmoide"""
    z = sum(x * w for x, w in zip(features, weights)) + bias
    return sigmoid(z)

#Algoritmo de entrenamiento (Descenso de Gradiente)
def train_logistic_regression(
        X: list[list[float]],
        y: list[int],
        learning_rate: float = 0.01,
        epochs: int = 1000
) -> tuple[list[float], float]:
    """ Entrena una regresion logistica desde cero ajustando pesos y sesgo."""
    num_samples = len(X)
    num_features = len(X[0])

    # Inicializamos pesos y sesgo
    weights = [0.0] * num_features
    bias = 0.0

    print(f"Iniciando entrenamiento desde cero ({epochs} epocas) ...")

    for epoch in range(epochs):
        dw = [0.0] * num_features
        db = 0.0

        # Calcular gradientes sobre todo el dataset
        for i in range(num_samples):
            y_pred = predict_probability(X[i], weights, bias)
            error = y_pred - y[i]

            for j in range(num_features):
                dw[j] += error * X[i][j]
            db += error

        # Actualizar parametros (Paso de aprendizaje)
        for j in range(num_features):
            weights[j] -= (learning_rate / num_samples) * dw[j]
        bias -= (learning_rate / num_samples) * db

        # Imprimir progreso cada 200 epocas
        if (epoch + 1) % 200 == 0 or epoch == 0:
            loss = -sum(
                y[i] * math.log(max(1e-15, predict_probability(X[i], weights, bias))) +
                (1 - y[i]) * math.log(max(1e-15, 1 - predict_probability(X[i], weights, bias)))
                for i in range(num_samples)
            ) / num_samples
            print(f"  Época {epoch + 1}/{epochs} - Pérdida (Loss): {loss:.4f}")

    return weights, bias

#Funcion principal del pipeline 
def run_custom_training(data_path: str, split_ratio: float = 0.8):
    print(f"Cargando dataset procesado desde '{data_path}'...")
    df = pd.read_parquet(data_path)

    feature_cols = ["Return", "SMA_5", "SMA_20", "Volatility_5"]

    # Normalizacion Min - Max desde cero (Crucial para algoritmos de gradiente)
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)

    X_list = df[feature_cols].values.tolist()
    y_list = df["Target"].tolist()

    # Division de los datos en Train / Test
    split_index = int(len(df) * split_ratio)
    X_train, X_test = X_list[:split_index], X_list[split_index:]
    y_train, y_test = y_list[:split_index], y_list[split_index:]

    # Entrenar modelo manual
    weights, bias = train_logistic_regression(X_train, y_train, learning_rate=0.1, epochs=1000)

    # Evaluacion del conjunto de prueba (Test)
    correct_predictions = 0
    for i in range(len(X_test)):
        prob = predict_probability(X_test[i], weights, bias)
        prediction = 1 if prob >= 0.5 else 0
        if prediction == y_test[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(X_test)

    print("\n--- Resultados del modelo manual desde cero ---")
    print(f"Pesos Aprendidos (Weights): {[round(w, 4) for w in weights]}")
    print(f"Sesgo Aprendido (Bias):     {bias:.4f}")
    print(f"Exactitud en Test (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    DATA_SET = "data/processed/sp500_caracteristicas.parquet"
    run_custom_training(data_path=DATA_SET)