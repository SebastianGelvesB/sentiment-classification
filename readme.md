# Clasificación de Sentimientos de Tweets con Modelos Transformer

---

![Mercado Libre Logo](./assets/meli_logo.png)

Este proyecto fue desarrollado como parte de un **challenge técnico para la posición de Data Scientist en Mercado Libre**.


Consiste en implementar un modelo de clasificación de sentimientos utilizando arquitecturas Transformer preentrenadas — BERT, RoBERTa y DistilBERT — ajustadas mediante fine-tuning sobre un conjunto de datos de tweets relacionados con COVID-19.

El objetivo es clasificar cada tweet en una de las siguientes cinco categorías de sentimiento:

- Extremadamente negativo  
- Negativo  
- Neutral  
- Positivo  
- Extremadamente positivo

---


## Estructura del proyecto

El proyecto está estructurado de la siguiente manera:

- `data`: En este folder se encuentran los datos raw, clean y procesados.

- `scripts`: Se encuentra el script `02_preprocess.py`que orquesta el preprocesamiento de los datos

- `src`: Recopila los scripts desarrollados para llevar a cabo el procesamiento de datos de forma moludarizada

- `notebooks`: Se encuentran los notebooks: 

    *   `01_eda.ipynb`: Se desarrolla todo el EDA del dataset crudo
    *   `03_modeling.ipynb`: Se desarrolla todo el modelado estructurado de 3 diferentes modelos transformers (BERT, RoBERTa y DistilBERT). Adicionalmente se hace la respectiva evaluación de cada modelo y una comparación entre si.

- `outputs`: Se encuentra el archivo `resultados_experimentos.json`, el cual lleva un tracking de todos los modelos que se entrenen y sus métricas.

- `models`: En esta carpeta se guardan los modelos entrenados, siempre y cuando se fije `save_model=True`

---


## Instrucciones para ejecutar este proyecto

1. Instale los requerimientos en `requirements.txt`

2. Ejecute y lea el notebook  `01_eda.ipynb` para entender la estructura de los datos crudos.

3. Corra el script `02_preprocess.py` para ejecutar el preprocesamiento de datos.

4. Ejecute el notebook `03_modeling.ipynb` para entrenar los modelos predeterminados por los hiperparámetros en el notebook, o si lo desea, cambie estos para costumizar los modelos.

5. Si fija `save_model=True`, podra acceder a los modelos guardados para su posterior uso.

---


## Documentación del modelo

A continuación, se describen los pasos principales del flujo de modelado, las funciones implementadas y su justificación técnica.



### Paso 1: Tokenización del texto

**Funciones involucradas:**
- `tokenize_data`
- `tokenize_datasets`

La tokenización transforma el texto crudo en representaciones numéricas compatibles con los modelos Transformer. Se utiliza un tokenizador específico para cada arquitectura (`AutoTokenizer.from_pretrained(model_name)`), con truncamiento, padding y una longitud máxima de secuencia (`max_length=128`), definida con base en el análisis exploratorio.




### Paso 2: Conversión de datos a tensores

**Funciones involucradas:**
- `convert_to_tensor`
- `create_tensordatasets`

Las etiquetas (`Sentiment`) se convierten en tensores, y junto con los `input_ids` y `attention_mask`, se agrupan en objetos `TensorDataset`, estructura de datos requerida para trabajar en PyTorch.




### Paso 3: Creación de DataLoaders

**Función involucrada:**
- `create_dataloader`

Se crean `DataLoader` para los datasets de entrenamiento, validación y prueba, permitiendo procesar los datos por lotes (`batch_size`) y barajarlos en el entrenamiento (`shuffle=True`).



### Paso 4: Definición de la función de pérdida

**Función involucrada:**
- `compute_class_weight`
- `calculate_loss_fn`

Se crea la función de pérdida para el modelo utilizando `CrossEntropyLoss` ponderada con pesos derivados de la frecuencia de clases en el dataset, usando `compute_class_weight` con la finalidad de tratar el desbalance de las clases.


---

### Paso 5: Entrenamiento del modelo (fine-tuning)

**Función involucrada:**
- `train_model`

Durante esta etapa se realiza el fine-tuning del modelo Transformer preentrenado utilizando los datos previamente procesados y estructurados. El entrenamiento se ejecuta por ciclos (`epochs`) sobre el conjunto de entrenamiento, con evaluación al final de cada época sobre el conjunto de validación. Este proceso consta de varias subetapas clave:



#### 1. Inicialización del modelo

Se utiliza la clase `AutoModelForSequenceClassification` de Hugging Face, que carga un modelo preentrenado (por ejemplo, BERT, RoBERTa o DistilBERT) y añade una capa densa de clasificación para tareas supervisadas multiclase, específicamente para 5 clases.


`model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=5,return_dict=True)`



#### 2. Definición del optimizador

Se define `AdamW`, una versión de Adam con weight decay desacoplado, como optimizador del modelo Transformer. Idealmente se debe hacer un análisis de Learning Rate Finding, para encontrar el hiperparámetro que permite obtener el punto donde el loss decrece más rápido, antes de que estalle.

`optimizer = AdamW(model.parameters(),lr=lr, weight_decay=weight_decay)`



#### 3. Definimos el scheduler 

Definimos el scheduler que permite controlar dinámicamente el `learning_rate` a lo largo del entrenamiento.
````
scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=epochs * len(train_dataloader)
)
````


#### 4. Ciclo de entrenamiento y evaluación por epochs

El modelo se entrena por epochs. En cada epoch se itera por batches sobre el conjunto de entrenamiento. Se realiza forward pass, cálculo de la pérdida, retropropagación, actualización de pesos y scheduler step. Luego, se evalúa el desempeño sobre el conjunto de validación.

En detalle:

- `` model.train()`` activa los componentes de entrenamiento como dropout.

- ``zero_grad()`` previene acumulación de gradientes de pasos anteriores.

- El forward pass produce los logits.

- La función de pérdida calcula el error entre las predicciones y los targets.

- ``backward()`` propaga el gradiente y ``step()`` actualiza los pesos.

- El scheduler ajusta la tasa de aprendizaje tras cada batch.

---


### Paso 6: Guardado del modelo y tokenizer

**Función involucrada:**
- `save_model_tokenizer`

Se guardan el modelo y el tokenizer en la ruta especificada, utilizando los métodos `save_pretrained`.



### Paso 7: Evaluación del modelo

**Funciones involucradas:**
- `evaluate_model`
- `save_experiment_results`

Se calcula el `accuracy`, `F1 macro` y el `classification_report`. Los resultados se visualizan con una matriz de confusión y se guardan en un archivo `.json` para tener la trazabilidad de los experimentos realizados.




### Paso 8: Ejecución y orquestación del pipeline

**Función involucrada:**
- `execute_modeling_pipeline`

Esta función orquesta todos los pasos anteriores en una ejecución secuencial. Permite entrenar y evaluar un modelo completo con una llamada por cada una.


---

## Conclusiones de los modelos


### BERT


- Al observar la evolución del entrenamiento del modelo a lo largo de los tres epochs, notamos que tanto `loss` como el `accuracy` mejoran consistentemente con cada iteración. Esto es una señal clara de que el modelo está aprendiendo de manera efectiva y, de momento, no estamos experimentando un underfitting.

- Sin embargo, hay un detalle importante a considerar: en el último epoch, el valor de pérdida en el conjunto de validación muestra un ligero aumento en comparación con el epoch anterior, mientras que el valor de pérdida en el conjunto de entrenamiento sigue disminuyendo. Esta divergencia sugiere que podríamos estar adentrándonos en una zona de overfitting. 

- Al analizar la evaluación en el data de test del modelo BERT después de haberle realizado el proceso de fine-tuning, se obtienen resultados excelentes:

    - `accuracy promedio = 0.85`: Esto implica que en el 85% de las predicciones realizadas sobre datos *no observados*, el modelo clasificó correctamente los tweets en su respectiva clase de `Sentiment`.

    - `F1-score macro = 0.86`, `F1-score weighted = 0.85`: Estos valores indican un rendimiento sólido y consistente en todas las clases de `Sentiment`. Esto evidencia la efectidad de la función de pérdida ponderada cálculada, ya que equilibra el aprendizaje del modelo preveniendo sesgos por las clases mayoritarias.

En resumen, el fine-tuning del modelo BERT presenta un performance muy bueno, demostrado por su alta capacidad de clasificar con alta precisión la clase de `Sentiment` de cada tweet.



### RoBERTa


### DistilBERT