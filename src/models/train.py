import pathlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score

def modelo_entrenamiento_linea_base(
        data_path: str, split_ratio: float = 0.8
) -> tuple[RandomForestClassifier, dict]:
    """Lee el dataset procesado, y realiza una division para entrenamiento y prueba,
    
    despues entrena un modelo de clasificacion Random Forest """
    print(f"XXXWWW===> Cargando dataset desde: {data_path} <===WWWXXX")
    df = pd.read_parquet(data_path)

    # Definir las caracteristicas (X) y la variable objetivo (y)
    feature_cols = ["Return", "SMA_5", "SMA_20", "Volatility_5"]
    X = df[feature_cols]
    y = df["Target"]

    # Division temporal del dataset en Train / Test, sin mezclar datos (Shuffle=False) para evitar data leakage
    split_index = int(len(df) * split_ratio)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(
        f"Datos de entrenamiento (Train): {len(X_train)} filas || Datos de evaluacion (Test): {len(X_test)} filas"
    )

    # Vamos a inicializar y entrenar el modelo
    # (random_state asegura su reproducibilidad en MLOps)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluacion del modelo en los datos de prueba
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    metrics = {"Exactitud": accuracy, "Precision": precision}

    print(f"""
    --- Resultados de la evaluacion en Test ---
    Exactitud (Accuracy): {accuracy:.4f}
    Precision (Precision): {precision:.4f}
    Reporte detallado de la clasificacion:
    """)
    print(classification_report(y_test, y_pred))

    return model, metrics

if __name__ == "__main__":
    DATA_SET = "data/processed/sp500_caracteristicas.parquet"
    modelo_entrenamiento_linea_base(data_path=DATA_SET)





