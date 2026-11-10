import datetime
import pandas as pd
import json
import pathlib
import requests

def get_raw_financial_json(
        symbol: str, began_timestamp: int, end_timestamp: int
) -> dict: 
    "Enviamos una solicitud a la API de Yahoo Finance para obtener los datos financieros en formato JSON."

    #URL de la API de Yahoo Finance
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    # Parámetros de la solicitud, los que se buscan en el url
    params = { 
        #fechas de inicio y fin en formato unix timestamp (segundos)
        "period1": began_timestamp,
        "period2": end_timestamp,
        "interval": "1d",  # Intervalo de los datos, en este caso diario
        "events": "history",

    }

    #Encabezados HTTP para evitar que la solicitud sea bloqueada por la API
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    print(f"Enviando peticion HTTP GET a la URL: {url} con los parametros: {params}")
    response = requests.get(url, headers=headers, params=params)

    # Verificamos si la solicitud fue exitosa
    response.raise_for_status()

    # Devolvemos los datos en formato JSON
    return response.json()


def parsear_json_a_dataframe(json_data: dict) -> pd.DataFrame:
    "Extrae las listas numericas de json y las convierte en una DataFrame estructurado"
    try:
        result = json_data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]

        #Estructuramos los datos en un DataFrame
        df = pd.DataFrame(
            {
                "Timestamps": timestamps,
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Volume": quote.get("volume"),
        }
        )

        # Convertimos el Timestamp unix (segundos desde 1970) a fechas legibles (YYYY-MM-DD)
        df["Date"] = pd.to_datetime(df["Timestamps"], unit="s")
        df.set_index("Date", inplace=True)
        df.drop(columns=["Timestamps"], inplace=True)

        # Limpiamos el DataFrame de dias festivos o nulos
        df.dropna(inplace=True)

        return df
    
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"Error al procesar la estructura del JSON recibido: {e}"
        )


def get_unix_timestamp(date_str: str) -> int:
    "Convierte una cadena de fecha en formato 'YYYY-MM-DD' a un timestamp Unix"
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())

def descarga_datos_financieros(
        symbol: str, began_date: str, end_date: str, output_path: str, outpath_de_prueba: str 
):
    began_ts = get_unix_timestamp(began_date)
    end_ts = get_unix_timestamp(end_date)

    # Obtenemos el JSON mediante peticion HTTP a la API de Yahoo Finance
    raw_json = get_raw_financial_json(symbol, began_ts, end_ts)

    # Convertimos el JSON a un DataFrame estructurado
    df = parsear_json_a_dataframe(raw_json)

    # Guardamos el DataFrame en formato parquet
    destination = pathlib.Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination)

    print(f"¡Exito! {len(df)} registros guardados en '{destination.resolve()}'")

    # Guardamos una copia de prueba en csv
    test_destination = pathlib.Path(outpath_de_prueba)
    test_destination.parent.mkdir(parents=True, exist_ok=True)
    df.head(10).to_csv(test_destination)

if __name__ == "__main__":
    # Configuración de prueba inicial
    descarga_datos_financieros(
        symbol = "^GSPC",
        began_date = "2020-01-01",
        end_date = "2026-01-01",
        output_path = "data/raw/sp500_diario.parquet",
        outpath_de_prueba = "data/raw/vista_previa_muestra.csv"
    )