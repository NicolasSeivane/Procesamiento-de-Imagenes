# Procesamiento de Imágenes / Image Processing

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Advanced Image Processing Library with Multiple Interfaces**

[English](#english) | [Español](#español)

</div>

---

## Español

### 📋 Descripción

Biblioteca completa de procesamiento de imágenes desarrollada como proyecto académico, implementando algoritmos avanzados desde cero para filtrado, detección de bordes, extracción de características y análisis de ruido. El proyecto incluye tres versiones diferentes para demostrar versatilidad en el desarrollo:

- **Versión 1 (main)**: Interfaz gráfica con Tkinter optimizada con Numba
- **Versión 2 (opencv-version)**: Implementación usando funciones nativas de OpenCV
- **Versión 3 (streamlit-version)**: Aplicación web interactiva con Streamlit

### ✨ Características Principales

#### Operaciones Básicas
- ✅ Ecualización de histograma
- ✅ Transformaciones gamma
- ✅ Umbralización (fija e iterativa)
- ✅ Operaciones aritméticas entre imágenes
- ✅ Conversión a escala de grises

#### Filtros y Máscaras
- 🔲 Filtro de media
- 🔲 Filtro de mediana (ponderada y estándar)
- 🔲 Filtro gaussiano
- 🔲 Filtro bilateral
- 🔲 Difusión anisotrópica (Leclerc, Lorentz)
- 🔲 Filtro de realce
- 🔲 Kernels personalizables

#### Detección de Bordes
- 🔍 Operadores Prewitt y Sobel
- 🔍 Laplaciano
- 🔍 Marr-Hildreth (LoG)
- 🔍 Canny
- 🔍 SUSAN (bordes y esquinas)

#### Análisis Avanzado
- 🎯 Algoritmo SIFT (Scale-Invariant Feature Transform)
- 🎯 Transformada de Hough para detección de líneas
- 🎯 Umbralización de Otsu
- 🎯 Detección de cruces por cero

#### Generación y Análisis de Ruido
- 📊 Ruido Gaussiano
- 📊 Ruido Rayleigh
- 📊 Ruido Exponencial
- 📊 Ruido Sal y Pimienta
- 📊 Visualización de histogramas

### 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/NicolasSeivane/Procesamiento-de-Imagenes.git
cd Procesamiento-de-Imagenes

# Instalar dependencias
pip install -r requirements.txt
```

### 💻 Uso

#### Versión 1: Interfaz Tkinter (Recomendada)
```bash
python main.py
```

#### Versión 2: OpenCV Nativo
```bash
git checkout opencv-version
python main.py
```

#### Versión 3: Aplicación Web Streamlit
```bash
git checkout streamlit-version
streamlit run streamlit_app.py
```

### 📁 Estructura del Proyecto

```
Procesamiento-de-Imagenes/
├── src/                          # Código fuente modular
│   ├── core/                     # Algoritmos principales
│   │   ├── image_processing.py   # Funciones de procesamiento
│   │   └── sift.py               # Implementación SIFT
│   ├── gui/                      # Interfaces gráficas
│   │   └── tkinter_app.py        # Aplicación Tkinter
│   └── utils/                    # Utilidades
│       └── noise_generator.py    # Generación de ruido
├── docs/                         # Documentación académica
│   ├── TP1_Procesamiento_de_Imagenes.pdf
│   ├── TP2_Procesamiento_de_Imagenes.pdf
│   ├── TP3_Procesamiento_de_Imagenes.pdf
│   └── TP4_Procesamiento_de_Imagenes.pdf
├── images/                       # Recursos de imágenes
│   ├── raw/                      # Imágenes originales
│   ├── contaminated/             # Imágenes con ruido
│   ├── tp3/                      # Imágenes TP3
│   └── examples/                 # Screenshots para README
├── examples/                     # Ejemplos de uso
├── main.py                       # Punto de entrada principal
├── requirements.txt              # Dependencias
└── README.md                     # Este archivo
```

### 🛠️ Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje principal
- **OpenCV**: Procesamiento de imágenes
- **NumPy**: Operaciones numéricas
- **Matplotlib**: Visualización
- **Tkinter**: Interfaz gráfica de escritorio
- **Streamlit**: Interfaz web interactiva
- **Numba**: Optimización de rendimiento (compilación JIT a C)
- **Pillow**: Manipulación de imágenes

### 📚 Contexto Académico

Este proyecto fue desarrollado como parte del curso de **Procesamiento de Imágenes** en la universidad. Los trabajos prácticos (TPs) documentan la evolución del proyecto:

- **TP1**: Operaciones básicas y transformaciones
- **TP2**: Filtros y máscaras
- **TP3**: Detección de bordes y características
- **TP4**: Algoritmos avanzados (SIFT, Hough)

Toda la documentación técnica está disponible en la carpeta [`docs/`](./docs/).

### 🎯 Características Destacadas para CV

- ✅ **Implementación desde cero**: Algoritmos implementados manualmente para demostrar comprensión profunda
- ✅ **Código modular**: Arquitectura limpia y mantenible
- ✅ **Múltiples interfaces**: Demuestra versatilidad (Desktop, Web, CLI)
- ✅ **Optimización**: Uso de Numba para acelerar código Python
- ✅ **Documentación completa**: README profesional y documentación académica
- ✅ **Control de versiones**: Uso apropiado de Git branches

### 📸 Capturas de Pantalla

<!-- 
TODO: Agregar capturas de pantalla aquí
Sugerencias de imágenes a incluir:

1. Screenshot de la interfaz principal de Tkinter
   - Guardar en: images/examples/tkinter_main_interface.png
   - Mostrar: Ventana principal con imagen cargada y controles visibles
   
2. Ejemplo de procesamiento (antes/después)
   - Guardar en: images/examples/before_after_processing.png
   - Mostrar: Comparación lado a lado de imagen original vs procesada
   
3. Demo de filtros
   - Guardar en: images/examples/filters_demo.png
   - Mostrar: Diferentes filtros aplicados a la misma imagen
   
4. Detección de bordes
   - Guardar en: images/examples/edge_detection.png
   - Mostrar: Resultados de Sobel, Canny, SUSAN

5. GIF animado del flujo de trabajo
   - Guardar en: images/examples/workflow_demo.gif
   - Mostrar: Proceso completo desde carga hasta descarga
   - Herramienta sugerida: ScreenToGif, LICEcap, o Peek

Formato de inserción:
![Descripción](./images/examples/nombre_archivo.png)

Para GIFs:
![Demo Animado](./images/examples/workflow_demo.gif)
-->

**Interfaz Principal**
<!-- ![Interfaz Tkinter](./images/examples/tkinter_main_interface.png) -->

**Procesamiento de Imágenes**
<!-- ![Antes y Después](./images/examples/before_after_processing.png) -->

**Detección de Bordes**
<!-- ![Detección de Bordes](./images/examples/edge_detection.png) -->

**Demo Interactiva (GIF)**
<!-- ![Workflow Demo](./images/examples/workflow_demo.gif) -->

*[Screenshots serán agregados próximamente]*

### 🤝 Contribuciones

Este es un proyecto académico, pero sugerencias y mejoras son bienvenidas. Por favor, abre un issue o pull request.

### 📧 Contacto

**Nicolas Seivane**
- GitHub: [@NicolasSeivane](https://github.com/NicolasSeivane)
- LinkedIn: [Tu perfil de LinkedIn]

### 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## English

### 📋 Description

Comprehensive image processing library developed as an academic project, implementing advanced algorithms from scratch for filtering, edge detection, feature extraction, and noise analysis. The project includes three different versions to demonstrate development versatility:

- **Version 1 (main)**: Tkinter GUI optimized with Numba
- **Version 2 (opencv-version)**: Implementation using native OpenCV functions
- **Version 3 (streamlit-version)**: Interactive web application with Streamlit

### ✨ Key Features

#### Basic Operations
- ✅ Histogram equalization
- ✅ Gamma transformations
- ✅ Thresholding (fixed and iterative)
- ✅ Arithmetic operations between images
- ✅ Grayscale conversion

#### Filters and Masks
- 🔲 Mean filter
- 🔲 Median filter (weighted and standard)
- 🔲 Gaussian filter
- 🔲 Bilateral filter
- 🔲 Anisotropic diffusion (Leclerc, Lorentz)
- 🔲 Enhancement filter
- 🔲 Customizable kernels

#### Edge Detection
- 🔍 Prewitt and Sobel operators
- 🔍 Laplacian
- 🔍 Marr-Hildreth (LoG)
- 🔍 Canny
- 🔍 SUSAN (edges and corners)

#### Advanced Analysis
- 🎯 SIFT Algorithm (Scale-Invariant Feature Transform)
- 🎯 Hough Transform for line detection
- 🎯 Otsu's thresholding
- 🎯 Zero-crossing detection

#### Noise Generation and Analysis
- 📊 Gaussian noise
- 📊 Rayleigh noise
- 📊 Exponential noise
- 📊 Salt and Pepper noise
- 📊 Histogram visualization

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/NicolasSeivane/Procesamiento-de-Imagenes.git
cd Procesamiento-de-Imagenes

# Install dependencies
pip install -r requirements.txt
```

### 💻 Usage

#### Version 1: Tkinter Interface (Recommended)
```bash
python main.py
```

#### Version 2: Native OpenCV
```bash
git checkout opencv-version
python main.py
```

#### Version 3: Streamlit Web App
```bash
git checkout streamlit-version
streamlit run streamlit_app.py
```

### 🛠️ Technologies Used

- **Python 3.8+**: Main language
- **OpenCV**: Image processing
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization
- **Tkinter**: Desktop GUI
- **Streamlit**: Interactive web interface
- **Numba**: Performance optimization (JIT compilation to C)
- **Pillow**: Image manipulation

### 📚 Academic Context

This project was developed as part of the **Image Processing** course at university. The practical assignments (TPs) document the project's evolution:

- **TP1**: Basic operations and transformations
- **TP2**: Filters and masks
- **TP3**: Edge and feature detection
- **TP4**: Advanced algorithms (SIFT, Hough)

All technical documentation is available in the [`docs/`](./docs/) folder.

### 🎯 CV Highlights

- ✅ **From-scratch implementation**: Algorithms manually implemented to demonstrate deep understanding
- ✅ **Modular code**: Clean and maintainable architecture
- ✅ **Multiple interfaces**: Demonstrates versatility (Desktop, Web, CLI)
- ✅ **Optimization**: Uses Numba to accelerate Python code
- ✅ **Complete documentation**: Professional README and academic documentation
- ✅ **Version control**: Proper use of Git branches

### 📧 Contact

**Nicolas Seivane**
- GitHub: [@NicolasSeivane](https://github.com/NicolasSeivane)
- LinkedIn: [Your LinkedIn profile]

### 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

</div>