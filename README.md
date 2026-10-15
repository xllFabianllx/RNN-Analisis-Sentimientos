\# Analizador de Sentimientos con Redes Neuronales Recurrentes (RNN)



\## 1. Descripción del proyecto



Este proyecto implementa un sistema de análisis de sentimientos para clasificar reseñas de películas como positivas o negativas mediante una red neuronal recurrente Bidirectional LSTM.



La aplicación cuenta con una interfaz desarrollada en Streamlit. El usuario escribe una reseña en español, el sistema la traduce al inglés y posteriormente el modelo realiza la predicción del sentimiento.



\## 2. Objetivo general



Desarrollar un sistema de clasificación de sentimientos utilizando aprendizaje profundo, capaz de analizar reseñas de películas y determinar si expresan una opinión positiva o negativa.



\## 3. Tecnologías utilizadas



\* Python.

\* TensorFlow y Keras.

\* NumPy.

\* Pandas.

\* Matplotlib.

\* Seaborn.

\* Scikit-learn.

\* Streamlit.

\* Requests.

\* Google Colab.



\## 4. Dataset



Se utilizó el conjunto de datos IMDB, compuesto por 50.000 reseñas de películas etiquetadas como positivas o negativas.



El conjunto de datos se distribuye en 25.000 reseñas para entrenamiento y 25.000 para prueba. A partir del conjunto original de entrenamiento se separaron datos para entrenamiento y validación.



El modelo fue entrenado con reseñas en inglés.



\## 5. Arquitectura del modelo



La red neuronal utiliza las siguientes capas:



1\. Embedding para representar las palabras mediante vectores.

2\. Bidirectional LSTM para procesar la secuencia de palabras en ambas direcciones.

3\. Dropout para reducir el sobreajuste.

4\. Dense con activación ReLU.

5\. Dense con activación sigmoide para realizar la clasificación binaria.



Se utilizó el optimizador Adam y la función de pérdida binary crossentropy.



\## 6. Resultados de evaluación



Los resultados obtenidos sobre el conjunto de prueba fueron:



| Métrica   | Resultado |

| --------- | --------: |

| Accuracy  |   83,89 % |

| Precision |   87,73 % |

| Recall    |   78,80 % |

| F1-score  |   83,02 % |



Estos resultados corresponden a la evaluación realizada durante el entrenamiento del modelo.



\## 7. Aplicación Streamlit



La aplicación permite:



\* Escribir reseñas en español.

\* Traducir las reseñas al inglés.

\* Clasificar el sentimiento como positivo o negativo.

\* Mostrar las probabilidades estimadas para ambas clases.



La traducción requiere conexión a Internet y disponibilidad del servicio externo utilizado.



\## 8. Estructura del proyecto



```text

RNN2/

├── entrenamiento.ipynb

├── app.py

├── requirements.txt

├── README.md

├── modelo/

│   ├── modelo\_lstm.keras

│   ├── modelo\_lstm.weights.h5

│   └── tokenizer.pkl

└── resultados/

&#x20;   ├── distribucion\_sentimientos.png

&#x20;   ├── matriz\_confusion.png

&#x20;   └── 10\_predicciones.csv

```



Los archivos de resultados deben incluirse cuando estén disponibles.



\## 9. Instalación y ejecución



\### Requisitos



Tener Python instalado y conexión a Internet para instalar las dependencias y utilizar el servicio de traducción.



\### Instalar dependencias



Desde la carpeta del proyecto, ejecutar:



```bash

python -m pip install -r requirements.txt

```



\### Ejecutar la aplicación



```bash

python -m streamlit run app.py

```



Streamlit abrirá la aplicación en el navegador.



\## 10. Archivos del modelo



\* `modelo\_lstm.weights.h5`: contiene los pesos entrenados utilizados por la aplicación.

\* `tokenizer.pkl`: contiene el vocabulario y los parámetros necesarios para procesar las reseñas.

\* `modelo\_lstm.keras`: contiene la versión completa del modelo guardado durante el entrenamiento.



La aplicación reconstruye la arquitectura y carga los pesos desde el archivo H5.



\## 11. Conclusión



El proyecto demuestra la aplicación de redes neuronales recurrentes para la clasificación de sentimientos en texto. La integración con Streamlit permite utilizar el modelo mediante una interfaz gráfica sencilla.



El modelo obtuvo una exactitud del 83,89 % sobre el conjunto de prueba. Su rendimiento puede mejorarse mediante ajustes en la arquitectura, los hiperparámetros y el preprocesamiento de los datos.



