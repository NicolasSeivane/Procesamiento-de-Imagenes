# Guía para Capturar Screenshots y GIFs

Esta guía te ayudará a crear las capturas de pantalla y GIFs necesarios para completar la documentación del proyecto.

## 📁 Estructura de Archivos

Todas las imágenes deben guardarse en:
```
images/examples/
```

## 🖼️ Screenshots Necesarios

### Para README.md Principal

#### 1. Interfaz Principal Tkinter
**Archivo**: `tkinter_main_interface.png`
**Qué capturar**:
- Ventana completa de la aplicación
- Imagen cargada visible
- Controles y botones visibles
- Panel de ROI si es posible

**Cómo**:
1. Ejecutar: `python main.py`
2. Cargar una imagen de prueba (usar `images/raw/`)
3. Capturar pantalla completa de la ventana
4. Guardar en `images/examples/tkinter_main_interface.png`

#### 2. Antes y Después
**Archivo**: `before_after_processing.png`
**Qué capturar**:
- Comparación lado a lado
- Imagen original vs procesada
- Mostrar una operación interesante (ej: detección de bordes)

**Cómo**:
1. Cargar imagen
2. Aplicar un filtro o detección de bordes
3. Capturar ambas imágenes visibles
4. Guardar en `images/examples/before_after_processing.png`

#### 3. Demo de Filtros
**Archivo**: `filters_demo.png`
**Qué capturar**:
- Diferentes filtros aplicados
- Puede ser un collage de 4 imágenes

**Cómo**:
1. Aplicar filtro gaussiano, mediana, etc.
2. Capturar cada resultado
3. Opcional: Usar herramienta de collage
4. Guardar en `images/examples/filters_demo.png`

#### 4. Detección de Bordes
**Archivo**: `edge_detection.png`
**Qué capturar**:
- Resultados de Sobel, Canny, SUSAN

**Cómo**:
1. Aplicar diferentes detectores
2. Capturar resultados
3. Guardar en `images/examples/edge_detection.png`

#### 5. GIF de Workflow
**Archivo**: `workflow_demo.gif`
**Qué capturar**:
- Proceso completo: cargar → procesar → guardar
- Duración: 10-15 segundos

**Herramientas**:
- **Windows**: ScreenToGif (https://www.screentogif.com/)
- **Mac**: Kap (https://getkap.co/)
- **Linux**: Peek (https://github.com/phw/peek)

**Cómo**:
1. Abrir herramienta de grabación
2. Iniciar grabación
3. Ejecutar workflow completo
4. Detener y exportar como GIF
5. Guardar en `images/examples/workflow_demo.gif`

---

### Para README_OPENCV.md

#### 6. Gráfico de Comparación de Rendimiento
**Archivo**: `performance_comparison.png`
**Qué crear**:
- Gráfico de barras comparando tiempos

**Cómo**:
```python
import matplotlib.pyplot as plt
import numpy as np

operations = ['Ecualización', 'Umbralización', 'Gaussiano 5x5', 'Gamma']
v1_times = [245, 189, 1250, 312]  # ms
v2_times = [4.8, 2.3, 28, 3.1]    # ms

x = np.arange(len(operations))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, v1_times, width, label='Versión 1 (Manual)', color='#ff7f0e')
bars2 = ax.bar(x + width/2, v2_times, width, label='Versión 2 (OpenCV)', color='#1f77b4')

# Agregar etiquetas de speedup
for i, (v1, v2) in enumerate(zip(v1_times, v2_times)):
    speedup = v1 / v2
    ax.text(i, max(v1, v2) * 1.1, f'{speedup:.0f}x', 
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Tiempo (ms)', fontsize=12)
ax.set_title('Comparación de Rendimiento: Versión 1 vs Versión 2', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(operations, fontsize=11)
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('images/examples/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Para README_STREAMLIT.md

#### 7. Página Home de Streamlit
**Archivo**: `streamlit_home.png`
**Qué capturar**:
- Página principal completa
- Demo rápida visible

**Cómo**:
1. Ejecutar: `streamlit run streamlit_app.py`
2. Esperar a que cargue
3. Capturar pantalla completa del navegador
4. Guardar en `images/examples/streamlit_home.png`

#### 8. Página de Filtros
**Archivo**: `streamlit_filters.png`
**Qué capturar**:
- Interfaz de filtros
- Sliders y controles visibles
- Resultado del filtro

**Cómo**:
1. Ir a página "Filtros"
2. Cargar imagen
3. Ajustar algún filtro
4. Capturar pantalla
5. Guardar en `images/examples/streamlit_filters.png`

#### 9. Página de Detección de Bordes
**Archivo**: `streamlit_edges.png`
**Qué capturar**:
- Comparación de detectores
- Controles visibles

**Cómo**:
1. Ir a página "Detección de Bordes"
2. Aplicar detector (ej: Canny)
3. Capturar pantalla
4. Guardar en `images/examples/streamlit_edges.png`

#### 10. Página de Operaciones Básicas
**Archivo**: `streamlit_basic.png`
**Qué capturar**:
- Operación con histogramas
- Preferiblemente ecualización

**Cómo**:
1. Ir a página "Operaciones Básicas"
2. Aplicar ecualización
3. Mostrar histogramas
4. Capturar pantalla
5. Guardar en `images/examples/streamlit_basic.png`

#### 11. GIF Demo Interactivo Streamlit
**Archivo**: `streamlit_demo.gif`
**Qué capturar**:
- Usuario ajustando sliders
- Cambios en tiempo real
- Duración: 10-15 segundos

**Cómo**:
1. Abrir ScreenToGif/Kap/Peek
2. Iniciar grabación del navegador
3. Cargar imagen
4. Ajustar varios sliders (ej: gamma, umbral)
5. Mostrar cambios en tiempo real
6. Detener y exportar
7. Guardar en `images/examples/streamlit_demo.gif`

---

## 🎨 Consejos de Calidad

### Para Screenshots (PNG)
- **Resolución**: Mínimo 1920x1080
- **Formato**: PNG (mejor calidad)
- **Tamaño**: Optimizar si es > 2MB
- **Contenido**: Asegurar que el texto sea legible

### Para GIFs
- **Duración**: 10-15 segundos máximo
- **FPS**: 15-20 fps (suficiente para demos)
- **Tamaño**: Optimizar a < 5MB
- **Resolución**: 1280x720 o 1920x1080
- **Herramientas de optimización**:
  - https://ezgif.com/optimize
  - https://www.iloveimg.com/compress-image/compress-gif

### Herramientas Recomendadas

#### Captura de Pantalla
- **Windows**: Win + Shift + S (Snipping Tool)
- **Mac**: Cmd + Shift + 4
- **Linux**: Flameshot, Spectacle

#### Grabación de GIF
- **Windows**: [ScreenToGif](https://www.screentogif.com/) ⭐ Recomendado
- **Mac**: [Kap](https://getkap.co/)
- **Linux**: [Peek](https://github.com/phw/peek)

#### Edición de Imágenes
- **GIMP**: Edición avanzada (gratis)
- **Paint.NET**: Simple y rápido (Windows)
- **Preview**: Mac nativo
- **ImageMagick**: Línea de comandos

---

## 📝 Checklist

Marca cuando completes cada captura:

### README.md Principal
- [ ] `tkinter_main_interface.png`
- [ ] `before_after_processing.png`
- [ ] `filters_demo.png`
- [ ] `edge_detection.png`
- [ ] `workflow_demo.gif`

### README_OPENCV.md
- [ ] `performance_comparison.png`

### README_STREAMLIT.md
- [ ] `streamlit_home.png`
- [ ] `streamlit_filters.png`
- [ ] `streamlit_edges.png`
- [ ] `streamlit_basic.png`
- [ ] `streamlit_demo.gif`

---

## 🔄 Actualizar READMEs

Una vez que tengas las imágenes:

1. **Descomentar las líneas** en los README:
   ```markdown
   <!-- ![Descripción](./images/examples/archivo.png) -->
   ```
   
   Cambiar a:
   ```markdown
   ![Descripción](./images/examples/archivo.png)
   ```

2. **Eliminar el texto temporal**:
   ```markdown
   *[Screenshots serán agregados próximamente]*
   ```

3. **Verificar** que las imágenes se vean correctamente en GitHub

---

## 🎯 Resultado Final

Cuando termines, tu carpeta `images/examples/` debe contener:

```
images/examples/
├── tkinter_main_interface.png
├── before_after_processing.png
├── filters_demo.png
├── edge_detection.png
├── workflow_demo.gif
├── performance_comparison.png
├── streamlit_home.png
├── streamlit_filters.png
├── streamlit_edges.png
├── streamlit_basic.png
└── streamlit_demo.gif
```

**Total**: 11 archivos (6 PNG + 2 GIF para README principal, 1 PNG para OpenCV, 4 PNG + 1 GIF para Streamlit)

---

## 💡 Tips Adicionales

1. **Usa imágenes de prueba interesantes**: Fotos con buenos contrastes y detalles
2. **Mantén consistencia**: Usa la misma imagen de prueba en varios screenshots
3. **Optimiza tamaños**: Usa herramientas de compresión
4. **Verifica en GitHub**: Asegúrate de que se vean bien en el README renderizado
5. **Nombres descriptivos**: Ya están definidos, no cambies los nombres

¡Buena suerte! 📸
