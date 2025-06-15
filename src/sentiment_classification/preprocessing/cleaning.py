import re

def clean_text(text: str) -> str:
    """
    Función para realizar limpieza de datos en un tweet. Elimina urls, menicones, hastags y demás.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)         # URLs
    text = re.sub(r"@\w+", "", text)                   # menciones
    text = re.sub(r"#\w+", "", text)                   # hashtags
    text = re.sub(r"[^\w\s]", "", text)                # signos de puntuación
    text = re.sub(r'\n', " ", text)                    # Espacios verticales 
    text = re.sub(r"\s+", " ", text).strip()           # múltiples espacios y espacios al inicio y/o final
    return text
