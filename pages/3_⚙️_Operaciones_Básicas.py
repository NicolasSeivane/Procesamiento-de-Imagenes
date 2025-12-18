"""
Página de Operaciones Básicas - Streamlit
========================================
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.core import image_processing as ip

st.set_page_config(page_title="Operaciones Básicas", page_icon="⚙️", layout="wide")

st.title("⚙️ Operaciones Básicas")

# Sidebar
with st.sidebar:
    st.header("📁 Cargar Imagen")
    uploaded_file = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.session_state['basic_image'] = image
        st.success("✅ Imagen cargada")

if 'basic_image' not in st.session_state:
    st.info("👈 Por favor, carga una imagen usando el panel lateral")
    st.stop()

image = st.session_state['basic_image']

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Ecualización", "🌓 Gamma", "🎯 Umbralización", "🔄 Negativo"])

with tab1:
    st.header("Ecualización de Histograma")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
        
        # Histograma original
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
        ax.set_title('Histograma Original')
        ax.set_xlabel('Nivel de Gris')
        ax.set_ylabel('Frecuencia')
        st.pyplot(fig)
    
    with col2:
        result = ip.ecualizacion(image)
        
        st.subheader("Imagen Ecualizada")
        st.image(result, use_column_width=True, clamp=True)
        
        # Histograma ecualizado
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(result.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax.set_title('Histograma Ecualizado')
        ax.set_xlabel('Nivel de Gris')
        ax.set_ylabel('Frecuencia')
        st.pyplot(fig)
        
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            "ecualizada.png",
            "image/png"
        )

with tab2:
    st.header("Corrección Gamma")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
        
        gamma = st.slider("Valor de Gamma", 0.1, 3.0, 1.0, 0.1)
        st.info(f"""
        **Gamma = {gamma}**
        - γ < 1: Aclara la imagen
        - γ = 1: Sin cambios
        - γ > 1: Oscurece la imagen
        """)
    
    with col2:
        result = ip.funcion_y_preview(image, gamma, grises=True, estandarizar=True)
        
        st.subheader(f"Imagen con γ = {gamma}")
        st.image(result, use_column_width=True, clamp=True)
        
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            f"gamma_{gamma}.png",
            "image/png"
        )

with tab3:
    st.header("Umbralización")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
        
        metodo = st.selectbox("Método", ["Manual", "Otsu"])
        
        if metodo == "Manual":
            umbral = st.slider("Umbral", 0, 255, 128)
        else:
            st.info("El método de Otsu calcula automáticamente el umbral óptimo")
    
    with col2:
        if metodo == "Manual":
            result = ip.funcion_umbral_preview(image, umbral, grises=True, estandarizar=True)
            st.subheader(f"Umbralizada (T = {umbral})")
        else:
            result, umbral_otsu = ip.umbralizacion_Otsu(image, grises=True)
            st.subheader(f"Otsu (T = {umbral_otsu})")
            st.success(f"Umbral calculado: {umbral_otsu}")
        
        st.image(result, use_column_width=True, clamp=True)
        
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            f"umbralizada_{metodo.lower()}.png",
            "image/png"
        )

with tab4:
    st.header("Negativo de Imagen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
    
    with col2:
        result = ip.fnegativo(image, grises=True)
        
        st.subheader("Imagen Negativa")
        st.image(result, use_column_width=True, clamp=True)
        
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            "negativa.png",
            "image/png"
        )
