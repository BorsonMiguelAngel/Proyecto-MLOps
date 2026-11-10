import pathlib
import pandas as pd

def crear_caracteristicas_financieras(
        input_path: str, output_path: str 
) -> pd.DataFrame:
    """ Lee el archivo con el dataset en formato parquet crudo, genera las caracteristicas financieras 
     y el Target, y lo guarda en un nuevo archivo en data/processed/  """

    print(f"###====>XXX Cargando datos crudos desde '{input_path}'...")
    df = pd.read_parquet(input_path)

    # Aseguramos que los datos esten en orden cronologico 
    df = df.sort_index()

    #Creamos los indicadores financieros (Features)
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Volatility_5"] = df["Return"].rolling(window=5).std()

    # Crear la variable objetivo (Target)
    # El precio de cierre de mañana sera mayor que el de hoy
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)

    # Eliminamos la columna auxiliar "Next_Close" para no tener data leakage (fuga de información)
    df.drop(columns=["Next_Close"], inplace=True)

    # Limpieza de valores nulos (NaN) generados por los desplazamientos y ventanas moviles
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(
        f"Se eliminaron {initial_rows - len(df)} filas con valores nulos (ventanas moviles/Target)."
    )

    # Guardar el dataset procesado 
    destination = pathlib.Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination)

    print(
        f" Dataset procesado con exito! ({len(df)} filas guardadas) en '{destination.resolve()}'"
    )

    return df 

if __name__ == "__main__":
    RAW_DATA = "data/raw/sp500_diario.parquet"
    PROCESSED_DATA = "data/processed/sp500_caracteristicas.parquet"

    crear_caracteristicas_financieras(input_path=RAW_DATA, output_path=PROCESSED_DATA)
