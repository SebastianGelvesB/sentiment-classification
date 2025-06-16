# Importamos librerías

import pandas as pd
from sentiment_classification.preprocessing.load import load_data
from sentiment_classification.preprocessing.cleaning import clean_text
from sentiment_classification.preprocessing.encoding import encode_labels_train,  encode_labels_test
from sentiment_classification.preprocessing.split_data import train_val_split
from sklearn.model_selection import train_test_split


# Leemos datos
df_train = load_data('data/raw/Corona_NLP_train.csv')
df_test = load_data('data/raw/Corona_NLP_test.csv')


### Data Cleaning ###

# Seleccionamos columnas a trabajar
df_train_clean = df_train[['OriginalTweet','Sentiment']].copy()
df_test_clean = df_test[['OriginalTweet','Sentiment']].copy()

# Hacemos la limpieza de datos
df_train_clean['OriginalTweet'] = df_train_clean['OriginalTweet'].apply(clean_text)
df_test_clean['OriginalTweet'] = df_test_clean['OriginalTweet'].apply(clean_text)

# Guardamos los datos limpios
df_train_clean.to_csv('data/clean/train_clean.csv', index=False)
df_test_clean.to_csv('data/clean/test_clean.csv', index=False)

# Mensaje informando que se han guardado los datos limpios
print('\n #### Los datos limpios han sido guardados en la carpeta /data/clean #### \n')


### Data Preprocessing ###

# Hacemos partición de train y validación
df_train_processed = df_train_clean.copy()
df_train_processed, df_val_processed = train_val_split(df_train_processed)

# Hacemos la codificación de Sentiment con un ordinal encoder
# Codificamos train
df_train_processed["Sentiment"], encoder = encode_labels_train(df_train_processed["Sentiment"], save_path='models/ordinal_encoder.pkl')

# Codificamos validation
df_val_processed["Sentiment"] = encode_labels_test(df_val_processed["Sentiment"], encoder_path="models/ordinal_encoder.pkl")

# Codificamos  test
df_test_processed = df_test_clean.copy()
df_test_processed["Sentiment"] = encode_labels_test(df_test_processed["Sentiment"], encoder_path="models/ordinal_encoder.pkl")



# Mensaje informando que se han codificado los datos y guardado el encoder
print('\n #### Los datos han sido codificados y el encoder se ha guardado en la carpeta /models #### \n')

# Guardamos datos processed
df_train_processed.to_csv('data/processed/train_encoded.csv', index=False)
df_val_processed.to_csv('data/processed/validation_encoded.csv', index=False)
df_test_processed.to_csv('data/processed/test_encoded.csv', index=False)

# Mensaje informando que se han guardado los datos encoded
print('\n #### Los datos con la columna Sentiment codificada han sido guardados en la carpeta /data/processed #### \n')

