from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import joblib

def encode_labels_train(y, save_path = None):
    """
    Función para codificar las etiquetas categóricas con OrdinalEncoder.
    """
    ordered_labels = [
        ["Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"]
    ]

    encoder = OrdinalEncoder(categories=ordered_labels, dtype=int)
    y_encoded = encoder.fit_transform(y.to_numpy().reshape(-1, 1)).astype(int).flatten()

    if save_path:
        joblib.dump(encoder, save_path)

    return y_encoded, encoder


def encode_labels_test(y, encoder_path):
    """
    Codifica las etiquetas del conjunto de test usando un OrdinalEncoder ya entrenado.
    """
    encoder = joblib.load(encoder_path)
    y_encoded = encoder.transform(y.to_numpy().reshape(-1, 1)).astype(int).flatten()
    return y_encoded


def decode_labels(y_encoded, encoder):
    # Devuelve etiquetas en texto
    return encoder.inverse_transform(np.array(y_encoded).reshape(-1, 1)).flatten()