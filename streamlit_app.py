"""
Procesamiento de Imágenes - Aplicación Web Streamlit
===================================================

Aplicación web interactiva para procesamiento de imágenes.
Versión 3.0 - Streamlit

Author: Nicolas Seivane
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.core import image_processing as ip

# Configuración de la página
st.set_page_config(
    page_title="Procesamiento de Imágenes",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🖼️ Procesamiento de Imágenes</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Image+Processing", use_column_width=True)
    st.markdown("---")
    st.markdown("### 📋 Navegación")
    st.markdown("""
    Usa el menú de arriba para navegar entre las diferentes funcionalidades:
    
    - **🏠 Home**: Página principal
    - **🎨 Filtros**: Aplicar filtros a imágenes
    - **🔍 Detección de Bordes**: Algoritmos de detección
    - **🎯 SIFT**: Extracción de características
    - **📊 Análisis de Ruido**: Generación y análisis
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ Información")
    st.info("""
    **Versión**: 3.0 (Streamlit)
    
    **Autor**: Nicolas Seivane
    
    **Tecnologías**:
    - Python 3.10+
    - OpenCV
    - Streamlit
    - NumPy
    """)

# Contenido principal
st.markdown("## 👋 Bienvenido")

st.markdown("""
Esta aplicación web te permite aplicar diversos algoritmos de procesamiento de imágenes
de forma interactiva y visual.

### 🚀 Características Principales

- **Operaciones Básicas**: Ecualización, gamma, umbralización
- **Filtros Avanzados**: Gaussiano, mediana, bilateral, anisotrópico
- **Detección de Bordes**: Sobel, Prewitt, Canny, SUSAN
- **Características SIFT**: Extracción y matching de puntos clave
- **Análisis de Ruido**: Generación y visualización de diferentes tipos de ruido

### 📖 Cómo Usar

1. **Selecciona una página** del menú superior
2. **Carga una imagen** usando el botón de carga
3. **Ajusta los parámetros** con los controles interactivos
4. **Visualiza los resultados** en tiempo real
5. **Descarga** la imagen procesada

### 🎯 Algoritmos Implementados

""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🔧 Operaciones Básicas")
    st.markdown("""
    - Ecualización de histograma
    - Corrección gamma
    - Umbralización
    - Negativo
    - Operaciones aritméticas
    """)

with col2:
    st.markdown("#### 🎨 Filtros")
    st.markdown("""
    - Filtro de media
    - Filtro de mediana
    - Filtro gaussiano
    - Filtro bilateral
    - Difusión anisotrópica
    """)

with col3:
    st.markdown("#### 🔍 Detección")
    st.markdown("""
    - Sobel / Prewitt
    - Laplaciano
    - Canny
    - SUSAN
    - Transformada de Hough
    """)

st.markdown("---")

# Demo rápida
st.markdown("## 🎬 Demo Rápida")

demo_col1, demo_col2 = st.columns(2)

with demo_col1:
    st.markdown("### Carga una Imagen")
    uploaded_file = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        # Leer imagen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, caption="Imagen Original", use_column_width=True)
        
        # Guardar en session state
        st.session_state['demo_image'] = image

with demo_col2:
    st.markdown("### Resultado Procesado")
    
    if 'demo_image' in st.session_state:
        # Opciones de procesamiento rápido
        operation = st.selectbox(
            "Selecciona una operación",
            ["Ninguna", "Ecualización", "Negativo", "Escala de Grises", "Blur Gaussiano"]
        )
        
        image = st.session_state['demo_image']
        
        if operation == "Ecualización":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = ip.ecualizacion(gray)
            st.image(result, caption="Imagen Ecualizada", use_column_width=True, clamp=True)
        
        elif operation == "Negativo":
            result = 255 - image  # Simple negativo usando NumPy
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Imagen Negativa", use_column_width=True)
        
        elif operation == "Escala de Grises":
            result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            st.image(result, caption="Escala de Grises", use_column_width=True, clamp=True)
        
        elif operation == "Blur Gaussiano":
            result = cv2.GaussianBlur(image, (15, 15), 0)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Blur Gaussiano", use_column_width=True)
        
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Sin Procesar", use_column_width=True)
        
        # Botón de descarga
        if operation != "Ninguna":
            st.download_button(
                label="📥 Descargar Imagen Procesada",
                data=cv2.imencode('.png', result)[1].tobytes(),
                file_name=f"procesada_{operation.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
    else:
        st.info("👆 Carga una imagen en la columna izquierda para ver el resultado")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Desarrollado con ❤️ usando Streamlit y OpenCV</p>
    <p>© 2025 Nicolas Seivane | 
    <a href='https://github.com/NicolasSeivane/Procesamiento-de-Imagenes' target='_blank'>GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
