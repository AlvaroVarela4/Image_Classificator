import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

categorias = ["Drink", "Food", "Inside", "Menu", "Outside"]

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('Modelo_Base_entrenado.keras')
    return model

model = load_model()

def classify_image(image, model, progress_bar, status_placeholder):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((244, 244)) 
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0

    # Actualizar el progreso en pasos
    status_placeholder.text("Starting prediction...")
    time.sleep(0.5)  # Simulamos un paso inicial
    progress_bar.progress(25)

    status_placeholder.text("Processing image...")
    time.sleep(0.5)  # Simulamos otro paso intermedio
    progress_bar.progress(50)

    status_placeholder.text("Generating predictions...")
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions, axis=1)[0]
    time.sleep(0.5)  # Simulamos el paso final
    progress_bar.progress(100)

    status_placeholder.empty()  # Limpiamos el marcador de posición
    
    return categorias[class_index]

def main():
    st.title("BiteNet Classificator")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("")
        if st.button("Classify Image"):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()  # Crear un marcador de posición
            with st.spinner('Classifying...'):
                result = classify_image(image, model, progress_bar, status_placeholder)
                st.write(f"Prediction: {result}")

if __name__ == "__main__":
    main()
