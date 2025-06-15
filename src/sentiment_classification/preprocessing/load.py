import pandas as pd

def load_data(path:str) -> pd.DataFrame:
    """
    Función para hacer la carga de datos desde un path
    """
    df = pd.read_csv(path, encoding='ISO-8859-1')
    return df

