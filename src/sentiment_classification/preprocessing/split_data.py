import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_split(df):
    '''
    Función para hacer partición del dataset de train en train y validation
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(
        df["OriginalTweet"],  
        df["Sentiment"],  
        test_size=0.2,  
        stratify=df["Sentiment"],  
        random_state=42
    )

    df_train = pd.DataFrame({'OriginalTweet':X_train, 'Sentiment':y_train})
    df_valid = pd.DataFrame({'OriginalTweet':X_valid, 'Sentiment':y_valid})

    return df_train, df_valid