import os
import re
import pickle
import requests
import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

RUTA_PESOS = "modelo/modelo_lstm.weights.h5"
RUTA_TOKENIZER = "modelo/tokenizer.pkl"

VOCAB_SIZE = 20000
MAX_LENGTH = 200


# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Analizador de sentimientos",
    page_icon="🎬",
    layout="centered"
)


# ==========================================================
# CARGAR MODELO
# ==========================================================

@st.cache_resource
def cargar_modelo():
    if not os.path.exists(RUTA_PESOS):
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo: {RUTA_PESOS}"
        )

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=64,
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    modelo.load_weights(RUTA_PESOS)

    return modelo


# ==========================================================
# CARGAR VOCABULARIO
# ==========================================================

@st.cache_resource
def cargar_vocabulario():
    if not os.path.exists(RUTA_TOKENIZER):
        raise FileNotFoundError(
            f"No se encontró el archivo del vocabulario: {RUTA_TOKENIZER}"
        )

    with open(RUTA_TOKENIZER, "rb") as archivo:
        datos = pickle.load(archivo)

    return (
        datos["word_index"],
        datos.get("vocab_size", VOCAB_SIZE),
        datos.get("max_length", MAX_LENGTH)
    )


# ==========================================================
# TRADUCIR ESPAÑOL A INGLÉS
# ==========================================================

def traducir_espanol_ingles(texto):
    """
    Traduce una reseña del español al inglés mediante MyMemory.
    Requiere conexión a Internet.
    """

    url = "https://api.mymemory.translated.net/get"

    parametros = {
        "q": texto,
        "langpair": "es|en"
    }

    respuesta = requests.get(
        url,
        params=parametros,
        timeout=20
    )

    respuesta.raise_for_status()

    datos = respuesta.json()

    if str(datos.get("responseStatus")) != "200":
        detalle = datos.get(
            "responseDetails",
            "El servicio de traducción no pudo procesar el texto."
        )
        raise ValueError(detalle)

    traduccion = datos.get("responseData", {}).get(
        "translatedText", ""
    )

    if not traduccion:
        raise ValueError("La traducción llegó vacía.")

    return traduccion


# ==========================================================
# PROCESAR TEXTO PARA EL MODELO
# ==========================================================

def convertir_texto_a_secuencia(texto, word_index, vocab_size):
    """
    Convierte el texto en números utilizando el vocabulario
    del conjunto de datos IMDB.
    """

    palabras = re.findall(
        r"\b[a-zA-Z0-9']+\b",
        texto.lower()
    )

    secuencia = [1]

    for palabra in palabras:
        indice = word_index.get(palabra, 2)

        if indice + 3 >= vocab_size:
            indice = 2
        else:
            indice += 3

        secuencia.append(indice)

    return secuencia


# ==========================================================
# INTERFAZ PRINCIPAL
# ==========================================================

st.title("🎬 Analizador de sentimientos")

st.write(
    """
    Escribe una reseña de una película en español.
    La aplicación la traducirá al inglés y utilizará una
    red neuronal recurrente **Bidirectional LSTM** para
    determinar si el sentimiento es positivo o negativo.
    """
)

st.info(
    "El modelo fue entrenado con reseñas de películas en inglés. "
    "Por eso, las reseñas en español se traducen antes de analizarlas."
)


# ==========================================================
# CARGAR RECURSOS
# ==========================================================

try:
    modelo = cargar_modelo()
    word_index, vocab_size, max_length = cargar_vocabulario()

    st.success("Modelo cargado correctamente.")

except Exception as error:
    st.error(f"No fue posible cargar el modelo: {error}")
    st.stop()


# ==========================================================
# ENTRADA DEL USUARIO
# ==========================================================

reseña = st.text_area(
    "Escribe aquí tu reseña:",
    placeholder=(
        "Ejemplo: La película estuvo excelente, "
        "los actores hicieron un gran trabajo."
    ),
    height=150
)


# ==========================================================
# ANALIZAR RESEÑA
# ==========================================================

if st.button("Analizar sentimiento", type="primary"):

    if not reseña.strip():
        st.warning("Por favor, escribe una reseña antes de analizarla.")

    else:
        try:
            # Traducir la reseña
            with st.spinner("Traduciendo la reseña al inglés..."):
                texto_ingles = traducir_espanol_ingles(reseña)

            st.subheader("Reseña traducida al inglés")
            st.write(texto_ingles)

            # Convertir texto a secuencia numérica
            secuencia = convertir_texto_a_secuencia(
                texto_ingles,
                word_index,
                vocab_size
            )

            # Ajustar la secuencia a 200 palabras
            secuencia_pad = pad_sequences(
                [secuencia],
                maxlen=max_length,
                padding="post",
                truncating="post"
            )

            # Realizar predicción
            with st.spinner("Analizando el sentimiento..."):
                probabilidad = float(
                    modelo.predict(
                        secuencia_pad,
                        verbose=0
                    )[0][0]
                )

            # Interpretar resultado
            confianza_positiva = probabilidad * 100
            confianza_negativa = (1 - probabilidad) * 100

            st.divider()
            st.subheader("Resultado del análisis")

            if probabilidad >= 0.5:
                st.success("😊 Sentimiento POSITIVO")
                st.write(
                    "El modelo considera que la reseña "
                    "expresa una opinión favorable."
                )
            else:
                st.error("😞 Sentimiento NEGATIVO")
                st.write(
                    "El modelo considera que la reseña "
                    "expresa una opinión desfavorable."
                )

            # Mostrar probabilidades
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Probabilidad positiva",
                    value=f"{confianza_positiva:.2f}%"
                )

            with col2:
                st.metric(
                    label="Probabilidad negativa",
                    value=f"{confianza_negativa:.2f}%"
                )

            st.write("Distribución de probabilidades")

            st.progress(
                max(0.0, min(1.0, probabilidad)),
                text=f"Positivo: {confianza_positiva:.2f}%"
            )

            st.caption(
                "La probabilidad representa la salida del modelo; "
                "no garantiza que la clasificación sea correcta."
            )

        except requests.exceptions.Timeout:
            st.error(
                "El servicio de traducción tardó demasiado. "
                "Inténtalo nuevamente."
            )

        except requests.exceptions.RequestException:
            st.error(
                "No fue posible conectarse al servicio de traducción. "
                "Verifica tu conexión a Internet e inténtalo nuevamente."
            )

        except Exception as error:
            st.error(
                f"No fue posible traducir o analizar la reseña: {error}"
            )


# ==========================================================
# INFORMACIÓN DEL PROYECTO
# ==========================================================

st.divider()

st.subheader("Información del proyecto")

st.markdown(
    """
    - **Tipo de problema:** clasificación de sentimientos.
    - **Modelo:** red neuronal recurrente Bidirectional LSTM.
    - **Dataset:** IMDB, con 50.000 reseñas de películas.
    - **Clases:** positivo y negativo.
    - **Idioma de entrada:** español.
    - **Idioma del modelo:** inglés.
    - **Interfaz:** Streamlit.
    """
)

st.caption(
    "Proyecto académico de inteligencia artificial y aprendizaje profundo."
)
