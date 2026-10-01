import pathlib
import pandas as pd
import yfinance as yf


def fetch_financial_data(
    ticker: str, start_date: str, end_date: str, output_path: str
) -> str:
    """Extrae datos financieros de mercado mediante yfinance y los guarda en el entorno cloud.

    Args:
        ticker (str): Símbolo del activo (ej. '^GSPC' para S&P 500, 'AAPL').
        start_date (str): Fecha de inicio en formato YYYY-MM-DD.
        end_date (str): Fecha de término en formato YYYY-MM-DD.
        output_path (str): Ruta donde se almacenará el dataset extraído.

    Returns:
        str: Ruta absoluta del archivo generado.
    """
    print(
        f"Iniciando descarga de datos para '{ticker}' desde {start_date} hasta {end_date}..."
    )

    # 1. Extracción desde la fuente pública
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(
            f"No se obtuvieron datos para el ticker '{ticker}'. Revisa el símbolo o las fechas."
        )

    # 2. Asegurar que las carpetas de destino existan
    destination = pathlib.Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 3. Guardar en formato Parquet
    data.to_parquet(destination)

    print(f" Éxito: {len(data)} registros guardados en '{destination.resolve()}'")
    return str(destination.resolve())


if __name__ == "__main__":
    # Configuración de prueba inicial
    TICKER = "^GSPC"  # Índice S&P 500
    START = "2020-01-01"
    END = "2026-01-01"
    SAVE_FILE = "data/raw/sp500_daily.parquet"

    fetch_financial_data(
        ticker=TICKER, start_date=START, end_date=END, output_path=SAVE_FILE
    )