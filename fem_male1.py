import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import requests
import os
import pickle
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from keras.preprocessing import image


def main():

    menu = ["Modelo CNN", "Introducción", "Muestra imagenes female", "Muestra imagenes male", "Código del modelo", "Gráficas", "Prueba"]

    page = st.sidebar.selectbox(label="Menu", options=menu)

    if page == "Modelo CNN":

        st.title("Convolutional Neural Networks (CNN) Female :woman-tipping-hand: Male :man-tipping-hand:")

        base_dir = '"C:\\Users\esthe\\Downloads\\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\'

        image = Image.open(
            "C:\\Users\esthe\\Downloads\\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\principal.jpg")

        st.image(image=image,
                 caption="pandas2",
                 use_column_width=True)
        st.write(
            "Una Red Neuronal Convolucional (Convolutional Neural Networks) tiene una estructura similar a un perceptrón multicapa; están formadas por neuronas que tienen parámetros en forma de pesos y biases.")
        st.write(
            "Las redes neuronales convolucionales están formadas de muchas capas CONVOLUCIONALES (CONV) y capas de submuestreo, conocidas como POOLING. Seguidas por una o más capas.")
        st.write(
            "La capa convolucional aprende patrones locales dentro de la imagen en pequeñas ventanas de 2 dimensiones.")
        st.write(
            "De forma general podemos decir que el propósito de la capa convolucional es detectar características o rasgos visuales en las imágenes que utiliza para el aprendizaje.")

        st.write("Estas características que aprende pueden ser: aristas, colores, formas, conjuntos de píxeles.")
        image1 = Image.open(
            "C:\\Users\esthe\\Downloads\\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\foto1.jpg")
        st.subheader(" Convolucion")
        st.image(image=image1,
                 caption="pandas2",
                 use_column_width=True)
        st.write(
            "De manera intuitiva, se puede decir que una capa convolucional es detectar características o rasgos visuales en las imágenes, como aristas, líneas, gotas de color, partes de una cara.")

        st.write(
            "Esto ayuda a que una vez que la red aprendió esta característica, la puede reconocer en cualquier imagen.")

        st.write(
            "En general las capas convolucionales operan sobre tensores 3D, llamados mapas de características (feature maps) donde se tienen las dimensiones de largo y ancho y una tercera que es el canal de las capas RGB.")
        st.subheader("Pooling")
        image2 = Image.open(
            "C:\\Users\\esthe\\Downloads\\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\foto.jpg")
        st.image(image=image2,
                 caption="pandas2",
                 use_column_width=True)
        st.write("Esta capa se suele aplicar inmediatamente después de la capa de convolución.")

        st.write(
            "Lo que hace la capa de pooling de manera simplificada es: reducir la información recogida por la capa convolucional y crean una versión condensada de la información contenida en esta capa.")

        # st.markdown("<h1 style='text-align: center; color: blue;'>Texto en <b>negrita</b></h1>", unsafe_allow_html=True)

    elif page == "Introducción":

        page_app_func()

        pass

    elif page == "Muestra imagenes female":



        # Define la función para mostrar las imágenes
        def mostrar_imagenes(nrows, ncols, train_female_dir):
            fig = plt.figure(figsize=(ncols * 4, nrows * 4))

            for i in range(nrows * ncols):
                random_female_number = random.randint(1, len(os.listdir(train_female_dir)))
                random_female = os.path.join(train_female_dir, os.listdir(train_female_dir)[random_female_number])

                # Agregar subplot
                sp = fig.add_subplot(nrows, ncols, i + 1)
                sp.axis('Off')

                # Leer la imagen
                img = mpimg.imread(random_female)
                plt.imshow(img)

            st.pyplot(fig)

        # Aplicación de Streamlit
        st.title('Aplicación con imágenes aleatorias')

        # Directorio de las imágenes de entrenamiento de mujeres
        train_female_dir = 'directorio_de_imagenes_de_entrenamiento_femeninas'

        # Parámetros para la disposición de las imágenes
        nrows = 4
        ncols = 4

        # Mostrar imágenes cuando se hace clic en el botón
        if st.button('Mostrar imágenes aleatorias'):
            mostrar_imagenes(nrows, ncols, train_female_dir)

        pass

    elif page == "Muestra imagenes male":

        ml()

        pass
    elif page == "Código del modelo":

        ml()

        pass
    elif page == "Gráficas":

        ml()

        pass
    elif page == "Prueba":


        # Cargar el modelo desde el archivo pickle
        with open('modelo.pkl', 'rb') as archivo:
            modelo = pickle.load(archivo)

        """def hacer_prediccion(datos):
            predicciones = modelo.predict(datos)
            return predicciones"""
        uploaded_file = st.sidebar.file_uploader(label="Sube una foto de cara", type=["jpg"])
        predicciones = []


        img = image.load_img(uploaded_file, target_size=(218, 178))
        img = image.img_to_array(img)
        img = tf.image.rgb_to_grayscale(img)
        img = np.expand_dims(img, axis=0)

        modelo = modelo.predict(img)

        predicciones.append(modelo)

        sp.axis('Off')

        plt.imshow(img.squeeze(), cmap='gray')

        plt.tight_layout()
        plt.show()

        if (modelo >= 0.5):
            print('Es un hombre')

        else:
            print('Es una mujer')
        # Aplicación de Streamlit
        st.title('Modelo CNN para determinar el sexo')
        


    pass




if __name__ == "__main__":
    main()


