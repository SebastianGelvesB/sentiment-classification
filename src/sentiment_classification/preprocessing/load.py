import pandas as pd

def load_data(path):
    """
    Función para hacer la carga de datos desde un path
    """
    df = pd.read_csv(path, encoding='ISO-8859-1')
    return df

