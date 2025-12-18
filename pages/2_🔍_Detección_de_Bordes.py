"""
Página de Detección de Bordes - Streamlit
========================================
"""

import streamlit as st
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.core import image_processing as ip

st.set_page_config(page_title="Detección de Bordes", page_icon="🔍", layout="wide")

st.title("🔍 Detección de Bordes")

# Sidebar
with st.sidebar:
    st.header("📁 Cargar Imagen")
    uploaded_file = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.session_state['edge_image'] = image
        st.success("✅ Imagen cargada")

if 'edge_image' not in st.session_state:
    st.info("👈 Por favor, carga una imagen usando el panel lateral")
    st.stop()

image = st.session_state['edge_image']

# Tabs para diferentes métodos
tab1, tab2, tab3, tab4 = st.tabs(["📐 Sobel", "🔲 Prewitt", "⚡ Canny", "🎯 SUSAN"])

with tab1:
    st.header("Detector de Bordes Sobel")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
    
    with col2:
        st.subheader("Sobel Horizontal")
        sobel_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        result_h = ip.mascara(image, sobel_h, "Sobel Horizontal", grises=True, estandarizar=True, prewitt=True)
        st.image(result_h, use_column_width=True, clamp=True)
    
    with col3:
        st.subheader("Sobel Vertical")
        sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        result_v = ip.mascara(image, sobel_v, "Sobel Vertical", grises=True, estandarizar=True, prewitt=True)
        st.image(result_v, use_column_width=True, clamp=True)
    
    # Magnitud combinada
    st.subheader("Magnitud del Gradiente")
    magnitude = np.sqrt(result_h.astype(np.float32)**2 + result_v.astype(np.float32)**2)
    magnitude = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255).astype(np.uint8)
    st.image(magnitude, use_column_width=True, clamp=True)
    
    st.download_button(
        "📥 Descargar Magnitud",
        cv2.imencode('.png', magnitude)[1].tobytes(),
        "sobel_magnitude.png",
        "image/png"
    )

with tab2:
    st.header("Detector de Bordes Prewitt")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
    
    with col2:
        st.subheader("Prewitt Horizontal")
        prewitt_h = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        result_h = ip.mascara(image, prewitt_h, "Prewitt Horizontal", grises=True, estandarizar=True, prewitt=True)
        st.image(result_h, use_column_width=True, clamp=True)
    
    with col3:
        st.subheader("Prewitt Vertical")
        prewitt_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        result_v = ip.mascara(image, prewitt_v, "Prewitt Vertical", grises=True, estandarizar=True, prewitt=True)
        st.image(result_v, use_column_width=True, clamp=True)

with tab3:
    st.header("Detector de Bordes Canny")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
        
        # Controles
        umbral1 = st.slider("Umbral Inferior", 0, 255, 50)
        umbral2 = st.slider("Umbral Superior", 0, 255, 150)
    
    with col2:
        # Aplicar Canny usando OpenCV directamente para mejor rendimiento
        result = cv2.Canny(image, umbral1, umbral2)
        
        st.subheader("Bordes Detectados")
        st.image(result, use_column_width=True, clamp=True)
        
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            "canny_edges.png",
            "image/png"
        )

with tab4:
    st.header("Detector SUSAN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True, clamp=True)
        
        # Controles
        umbral = st.slider("Umbral de Similitud", 1, 50, 15)
        tipo = st.selectbox("Tipo de Detección", ["borde", "esquina", "ambos"])
    
    with col2:
        result = ip.susan_bordes(image, umbral, tipo)
        
        st.subheader(f"Detección: {tipo.capitalize()}")
        st.image(result, use_column_width=True)
        
        st.download_button(
            "📥 Descargar",
            cv2.imencode('.png', result)[1].tobytes(),
            f"susan_{tipo}.png",
            "image/png"
        )

# Información adicional
with st.expander("ℹ️ Información sobre los Algoritmos"):
    st.markdown("""
    ### Sobel
    Utiliza kernels de convolución para calcular el gradiente de la imagen.
    Bueno para detectar bordes en direcciones específicas.
    
    ### Prewitt
    Similar a Sobel pero con pesos uniformes. Más sensible al ruido.
    
    ### Canny
    Algoritmo multi-etapa que incluye:
    1. Suavizado gaussiano
    2. Cálculo del gradiente
    3. Supresión no-máxima
    4. Umbralización con histéresis
    
    ### SUSAN
    (Smallest Univalue Segment Assimilating Nucleus)
    Detecta bordes y esquinas basándose en la similitud de intensidad local.
    """)
