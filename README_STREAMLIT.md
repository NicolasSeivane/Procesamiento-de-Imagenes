# Versión 3: Aplicación Web Streamlit

## 🌐 Descripción

Aplicación web interactiva para procesamiento de imágenes construida con Streamlit.

## ✨ Características

- 🖼️ **Interfaz Web Moderna**: UI intuitiva y responsive
- 📤 **Carga de Imágenes**: Drag & drop o selector de archivos
- 🎨 **Procesamiento en Tiempo Real**: Visualiza cambios instantáneamente
- 📊 **Visualización Interactiva**: Gráficos y comparaciones lado a lado
- 💾 **Descarga de Resultados**: Exporta imágenes procesadas
- 📱 **Responsive**: Funciona en desktop, tablet y móvil

## 🚀 Inicio Rápido

### Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run streamlit_app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

## 📄 Páginas Disponibles

<!--
TODO: Agregar screenshots de cada página de Streamlit

Sugerencias de capturas:
1. Página Home
   - Guardar en: images/examples/streamlit_home.png
   - Mostrar: Página principal con demo rápida
   
2. Página de Filtros
   - Guardar en: images/examples/streamlit_filters.png
   - Mostrar: Interfaz de filtros con controles interactivos
   
3. Página de Detección de Bordes
   - Guardar en: images/examples/streamlit_edges.png
   - Mostrar: Comparación de diferentes detectores de bordes
   
4. Página de Operaciones Básicas
   - Guardar en: images/examples/streamlit_basic.png
   - Mostrar: Ecualización con histogramas

5. GIF de interacción
   - Guardar en: images/examples/streamlit_demo.gif
   - Mostrar: Usuario ajustando sliders y viendo cambios en tiempo real
   - Duración sugerida: 10-15 segundos
   - Herramientas: ScreenToGif (Windows), Kap (Mac), Peek (Linux)

Formato de inserción:
![Nombre de Página](./images/examples/streamlit_pagina.png)

Para GIF:
![Demo Interactivo](./images/examples/streamlit_demo.gif)
-->

**Capturas de Pantalla**

<!-- ![Página Home](./images/examples/streamlit_home.png) -->
<!-- ![Filtros Interactivos](./images/examples/streamlit_filters.png) -->
<!-- ![Detección de Bordes](./images/examples/streamlit_edges.png) -->

**Demo Interactivo (GIF)**
<!-- ![Streamlit Demo](./images/examples/streamlit_demo.gif) -->

*[Screenshots de la aplicación web serán agregados próximamente]*

---

### 🏠 Home
- Introducción a la aplicación
- Demo rápida con operaciones básicas
- Información del proyecto

### ⚙️ Operaciones Básicas
- **Ecualización de Histograma**: Con visualización de histogramas
- **Corrección Gamma**: Control deslizante interactivo
- **Umbralización**: Manual y método de Otsu
- **Negativo**: Inversión de intensidades

### 🎨 Filtros
- **Filtro de Media**: Tamaño de kernel ajustable
- **Filtro Gaussiano**: Control de sigma
- **Filtro de Mediana**: Reducción de ruido
- **Difusión Anisotrópica**: Parámetros avanzados

### 🔍 Detección de Bordes
- **Sobel**: Horizontal, vertical y magnitud
- **Prewitt**: Detección de gradientes
- **Canny**: Umbralización dual interactiva
- **SUSAN**: Bordes, esquinas o ambos

## 🎨 Personalización

### Tema

Edita `.streamlit/config.toml` para cambiar colores:

```toml
[theme]
primaryColor = "#1f77b4"  # Color principal
backgroundColor = "#ffffff"  # Fondo
secondaryBackgroundColor = "#f0f2f6"  # Fondo secundario
textColor = "#262730"  # Color de texto
```

### Agregar Nueva Página

1. Crea un archivo en `pages/` con el formato: `N_🔥_Nombre.py`
2. El número `N` determina el orden
3. El emoji aparece en el menú
4. Usa la misma estructura que las páginas existentes

Ejemplo:

```python
# pages/4_🎯_Nueva_Funcionalidad.py
import streamlit as st

st.set_page_config(page_title="Nueva Funcionalidad", page_icon="🎯")
st.title("🎯 Nueva Funcionalidad")

# Tu código aquí...
```

## 📊 Estructura de Archivos

```
Procesamiento-de-Imagenes/
├── streamlit_app.py          # Aplicación principal
├── .streamlit/
│   └── config.toml           # Configuración y tema
└── pages/                    # Páginas multipágina
    ├── 1_🎨_Filtros.py
    ├── 2_🔍_Detección_de_Bordes.py
    └── 3_⚙️_Operaciones_Básicas.py
```

## 💡 Características Técnicas

### Session State
Streamlit usa `st.session_state` para mantener datos entre interacciones:

```python
# Guardar imagen
st.session_state['my_image'] = image

# Recuperar imagen
if 'my_image' in st.session_state:
    image = st.session_state['my_image']
```

### Caché
Funciones pesadas se pueden cachear:

```python
@st.cache_data
def procesar_imagen_pesada(img):
    # Procesamiento costoso
    return resultado
```

### Widgets Interactivos
- `st.slider()` - Control deslizante
- `st.selectbox()` - Menú desplegable
- `st.file_uploader()` - Carga de archivos
- `st.download_button()` - Descarga de resultados
- `st.tabs()` - Pestañas
- `st.columns()` - Layout en columnas

## 🎯 Casos de Uso

### 1. Demostración Académica
Ideal para presentaciones y clases:
- Visualización en tiempo real
- Comparación lado a lado
- Parámetros ajustables

### 2. Prototipado Rápido
Prueba algoritmos sin escribir GUI:
- Desarrollo iterativo
- Testing visual
- Feedback inmediato

### 3. Herramienta de Producción
Despliega en la nube:
- Streamlit Cloud (gratis)
- Heroku
- AWS/GCP/Azure

## 🚀 Despliegue

### Streamlit Cloud (Recomendado)

1. Sube tu código a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. ¡Listo! URL pública automática

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

```bash
docker build -t procesamiento-imagenes .
docker run -p 8501:8501 procesamiento-imagenes
```

## 📱 Responsive Design

La aplicación se adapta automáticamente a diferentes tamaños de pantalla:

- **Desktop**: Layout de 2-3 columnas
- **Tablet**: Layout de 2 columnas
- **Móvil**: Layout de 1 columna

## 🔧 Troubleshooting

### Error: "No module named 'src'"
```bash
# Asegúrate de ejecutar desde el directorio raíz
cd Procesamiento-de-Imagenes
streamlit run streamlit_app.py
```

### La aplicación es lenta
```python
# Usa @st.cache_data para funciones pesadas
@st.cache_data
def procesar_imagen(img):
    return resultado
```

### Imágenes no se muestran
```python
# Asegúrate de usar el formato correcto
st.image(imagen, use_column_width=True, clamp=True)
```

## 📚 Recursos

- [Documentación Streamlit](https://docs.streamlit.io/)
- [Galería de Apps](https://streamlit.io/gallery)
- [Foro de la Comunidad](https://discuss.streamlit.io/)
- [Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)

## 🎓 Para Aprender Más

### Tutoriales Recomendados
1. [Streamlit Basics](https://docs.streamlit.io/library/get-started)
2. [Building Multi-Page Apps](https://docs.streamlit.io/library/get-started/multipage-apps)
3. [Advanced Features](https://docs.streamlit.io/library/advanced-features)

## 🔗 Ver También

- [Versión 1 - Tkinter GUI](../README.md#version-1)
- [Versión 2 - OpenCV Optimizada](README_OPENCV.md)
- [Documentación Principal](../README.md)
