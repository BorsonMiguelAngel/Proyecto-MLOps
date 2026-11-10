import pathlib
import requests


def download_dataset(url: str, save_path: str) -> str:
    """Descarga un archivo desde una URL pública y lo almacena localmente en el entorno cloud.

    Args:
        url (str): Dirección web del recurso de datos.
        save_path (str): Ruta local dentro del contenedor/entorno donde se guardará.

    Returns:
        str: Ruta absoluta del archivo guardado.
    """
    path = pathlib.Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Valida que la respuesta HTTP sea 200 OK

    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Dataset descargado con éxito en: {path.resolve()}")
    return str(path.resolve())


if __name__ == "__main__":
    # Ejemplo con una fuente pública en formato Parquet (Viajes de Taxi NYC)
    DATASET_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    DESTINATION = "data/raw/yellow_tripdata_2024-01.parquet"

    download_dataset(DATASET_URL, DESTINATION)