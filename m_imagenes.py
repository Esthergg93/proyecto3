import streamlit as st
from PIL import Image



def m_imagenes():
    col1, col2 = st.columns([0.5, 0.5])

    image2 = Image.open(
        "C:\\Users\\esthe\\Downloads\\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\1.jpg")
    col1.image(image=image2, width=120)

    image1 = Image.open(
        "C:\\Users\\esthe\\Downloads\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\2.jpg")
    col2.image(image=image1, width=120)

    col3, col4 = st.columns([0.5, 0.5])
    image3 = Image.open(
        "C:\\Users\\esthe\\Downloads\\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\3.jpg")
    col3.image(image=image3, width=120)

    image4 = Image.open(
        "C:\\Users\\esthe\\Downloads\\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\4.jpg")
    col4.image(image=image4, width=120)

    col5, col6 = st.columns([0.5, 0.5])
    image5 = Image.open(
        "C:\\Users\\esthe\\Downloads\\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\5.jpg")
    col5.image(image=image5, width=120)

    image6 = Image.open(
        "C:\\Users\\esthe\\Downloads\\BOOTCAMP\\mod7-streamlit-master\\streamlit\\sources\\imagenes\\6.jpg")
    col6.image(image=image6, width=120)

if __name__ == "__main__":
    m_imagenes()
