"""
Página de Filtros - Streamlit
============================
"""

import streamlit as st
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.core import image_processing as ip

st.set_page_config(page_title="Filtros", page_icon="🎨", layout="wide")

st.title("🎨 Filtros de Imagen")

# Sidebar para carga de imagen
with st.sidebar:
    st.header("📁 Cargar Imagen")
    uploaded_file = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.session_state['filter_image'] = image
        st.success("✅ Imagen cargada")

if 'filter_image' not in st.session_state:
    st.info("👈 Por favor, carga una imagen usando el panel lateral")
    st.stop()

image = st.session_state['filter_image']

# Tabs para diferentes filtros
tab1, tab2, tab3, tab4 = st.tabs(["🔲 Filtro de Media", "📊 Filtro Gaussiano", "🎯 Filtro de Mediana", "✨ Difusión Anisotrópica"])

with tab1:
    st.header("Filtro de Media")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
    
    with col2:
        kernel_size = st.slider("Tamaño del Kernel", 3, 15, 5, step=2)
        
        # Aplicar filtro
        kernel = np.ones((kernel_size, kernel_size), np.float32)
        result = ip.mascara(image, kernel, "Media", grises=True, estandarizar=True)
        
        st.subheader("Imagen Filtrada")
        st.image(result, use_column_width=True, clamp=True)
        
        # Descarga
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            "filtro_media.png",
            "image/png"
        )

with tab2:
    st.header("Filtro Gaussiano")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
    
    with col2:
        sigma = st.slider("Sigma", 0.5, 5.0, 1.0, 0.1)
        kernel_size = int((2 * sigma) + 1)
        
        # Crear kernel gaussiano
        ax = np.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        result = ip.mascara(image, kernel, "Gaussiano", grises=True, estandarizar=True)
        
        st.subheader("Imagen Filtrada")
        st.image(result, use_column_width=True, clamp=True)
        
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            "filtro_gaussiano.png",
            "image/png"
        )

with tab3:
    st.header("Filtro de Mediana")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
    
    with col2:
        kernel_size = st.slider("Tamaño del Kernel (Mediana)", 3, 15, 5, step=2)
        
        kernel = np.ones((kernel_size, kernel_size), np.float32)
        result = ip.mascara(image, kernel, "Mediana", grises=True, estandarizar=True)
        
        st.subheader("Imagen Filtrada")
        st.image(result, use_column_width=True, clamp=True)
        
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            "filtro_mediana.png",
            "image/png"
        )

with tab4:
    st.header("Difusión Anisotrópica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
    
    with col2:
        t = st.slider("Iteraciones", 1, 50, 10)
        lambda_val = st.slider("Lambda", 0.01, 0.5, 0.25, 0.01)
        sigma = st.slider("Sigma (sensibilidad)", 1.0, 50.0, 10.0, 1.0)
        
        funcion = st.selectbox("Función", ["Leclerc", "Lorentz"])
        
        result = ip.anistropica(image, t, lambda_val, sigma, grises=True, estandarizar=True)
        
        st.subheader("Imagen Filtrada")
        st.image(result, use_column_width=True, clamp=True)
        
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            "difusion_anisotropica.png",
            "image/png"
        )
