# Clasificación de Sentimientos de Tweets con Modelos Transformer

Este proyecto fue desarrollado como parte de un **challenge técnico para la posición de Data Scientist en Mercado Libre**.

![Mercado Libre Logo](./assets/meli_logo.png)

Consiste en implementar un modelo de clasificación de sentimientos utilizando arquitecturas Transformer preentrenadas — BERT, RoBERTa y DistilBERT — ajustadas mediante fine-tuning sobre un conjunto de datos de tweets relacionados con COVID-19.

El objetivo es clasificar cada tweet en una de las siguientes cinco categorías de sentimiento:

- Extremadamente negativo  
- Negativo  
- Neutral  
- Positivo  
- Extremadamente positivo

El proyecto está estructurado de la siguiente manera:

- `data`: En este folder se encuentran los datos raw, clean y procesados.

- `scripts`: Se encuentra el script `02_preprocess.py`que orquesta el preprocesamiento de los datos

- `src`: Recopila los scripts desarrollados para llevar a cabo el procesamiento de datos de forma moludarizada

- `notebooks`: Se encuentran los notebooks: 

    *   `01_eda.ipynb`: Se desarrolla todo el EDA del dataset crudo
    *   `03_modeling.ipynb`: Se desarrolla todo el modelado estructurado de 3 diferentes modelos transformers (BERT, RoBERTa y DistilBERT). Adicionalmente se hace la respectiva evaluación de cada modelo y una comparación entre si.

- `outputs`: Se encuentra el archivo `resultados_experimentos.json`, el cual lleva un tracking de todos los modelos que se entrenen y sus métricas.

- `models`: En esta carpeta se guardan los modelos entrenados, siempre y cuando se fije `save_model=True`


## Instrucciones para ejecutar este proyecto

1. Instale los requerimientos en `requirements.txt`

2. Ejecute y lea el notebook  `01_eda.ipynb` para entender la estructura de los datos crudos.

3. Corra el script `02_preprocess.py` para ejecutar el preprocesamiento de datos.

4. Ejecute el notebook `03_modeling.ipynb` para entrenar los modelos predeterminados por los hiperparámetros en el notebook, o si lo desea, cambie estos para costumizar los modelos.

5. Si fija `save_model=True`, podra acceder a los modelos guardados para su posterior uso.