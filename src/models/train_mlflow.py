import pathlib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

def train_and_log_model(
        data_path: str, n_estimators: int = 100, max_depth: int = 5, split_ratio: float = 0.8
):
    """Entrena un modelo Random Forest y registra hiperparametros, metricas y el artefacto en Mlflow"""

    # 1. SOLUCIÓN AL ERROR DE PERMISOS: Forzar MLflow a usar una ruta local relativa dentro del proyecto
    mlruns_path = pathlib.Path("mlruns").resolve()
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    
    # 2. Configurar o seleccionar el experimento en Mlflow
    mlflow.set_experiment("SP500_Market_Prediction")

    print(f"Cargando dataset procesado desde '{data_path}'...")
    df = pd.read_parquet(data_path)

    feature_cols = ["Return", "SMA_5", "SMA_20", "Volatility_5"]
    X = df[feature_cols]
    y = df["Target"]

    # Division temporal Train / Test
    split_index = int(len(df) * split_ratio)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 2. Iniciar una corrida (Run) registrada por Mlflow
    with mlflow.start_run():
        print(f"Iniciando corrida en Mlflow (n_estimators={n_estimators}, max_depth={max_depth})...")

        # Registrar hiperparametros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", feature_cols)
        mlflow.log_param("data_path", data_path)

        # Entrenar modelo
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluar
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

        # Registrar metricas numericas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)

        # 3. Guardar el modelo fisico como artefacto de Mlflow
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        print("--- Corrida registrada con exito en Mlflow ---")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    Precision: {prec:.4f}")

if __name__ == "__main__":
    DATA_SET = "data/processed/sp500_caracteristicas.parquet"

    # Probamos una configuracion controlada (con limite de profundidad para Overfitting)
    train_and_log_model(data_path=DATA_SET, n_estimators=100, max_depth=3)
    train_and_log_model(data_path=DATA_SET, n_estimators=150, max_depth=5)