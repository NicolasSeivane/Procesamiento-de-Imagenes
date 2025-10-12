import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk
import funciones_principales as fc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==================
# Global State
# ==================
imagen_original = None      # BGR uint8
imagen_actual = None        # BGR/GRAY (la imagen sobre la que se trabaja)
roi_actual = None    
imagen_operativa = None  # copia de imagen_actual que usamos para previsualizaciones


# ==================
# Main Window
# ==================
root = tk.Tk()
root.title("Procesamiento de Imágenes - Editor de Píxeles Avanzado")
root.geometry("1200x800")
root.configure(bg="#2c3e50")  # Dark blue background
usar_roi = tk.BooleanVar(value=False)


# ttk Theme
style = ttk.Style(root)
style.theme_use("clam")  # Use 'clam' for a modern look
style.configure("TFrame", background="#2c3e50")
style.configure("TLabel", background="#2c3e50", foreground="#ecf0f1", font=("Helvetica", 11))
style.configure("TButton", background="#3498db", foreground="#ecf0f1", font=("Helvetica", 10, "bold"))
style.map("TButton",
          background=[("active", "#2980b9")],
          foreground=[("active", "#ecf0f1")])
style.configure("TScale", background="#2c3e50", troughcolor="#34495e")

# ==================
# Main Layout
# ==================
# Top section for file operations
frame_top = ttk.Frame(root, padding=10)
frame_top.pack(fill="x", pady=5)


# Image display section
frame_images = ttk.Frame(root, padding=10)
frame_images.pack(fill="both", expand=True)

# --- Zoom Slider ---
zoom_factor = 1.0
def set_zoom(val):
    global zoom_factor
    try:
        z = float(val)
    except Exception:
        return
    zoom_factor = z
    set_status(f"Zoom: {int(zoom_factor*100)}%")
    refrescar_imagenes()

frm_zoom = ttk.Frame(root, padding=8)
frm_zoom.pack(fill="x")
zoom_slider = ttk.Scale(frm_zoom, from_=0.1, to=3.0, orient='horizontal', command=set_zoom, length=400)
zoom_slider.set(1.0)
ttk.Label(frm_zoom, text="Zoom (factor)").pack(side="left", padx=8)
zoom_slider.pack(side="left", fill="x", expand=True)

def refrescar_imagenes():
    mostrar_imagen(imagen_original, lblInputImage, 300)
    mostrar_imagen(imagen_actual, lblOutputImage)
    mostrar_imagen(roi_actual, lblROI, 300)

# Processing controls section
frame_controls = ttk.Frame(root, padding=10)
frame_controls.pack(fill="x", pady=10)

# Status bar

lblStatus = ttk.Label(root, text="Listo.", anchor="w", relief="sunken", foreground="#ecf0f1", background="#34495e")
lblStatus.pack(fill="x", side="bottom")

def set_status(msg):
    lblStatus.config(text=msg)

# Mapping widget -> displayed size (width, height)
size_mostrada = {}


# ==================
# Utilities (from utilidades.py)
# ==================
def asegurar_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()

def guardar_imagen(imagen):
    if imagen is None:
        set_status("Nada para guardar.")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
    )
    if file_path:
        cv2.imwrite(file_path, imagen)
        set_status(f"Imagen guardada en {file_path}")

def mostrar_imagen(img, label, width_or_none=None):
    if img is None:
        label.config(image="")
        label.image = None
        return
    bgr = asegurar_bgr(img)
    # Apply zoom factor
    if width_or_none is not None:
        target_width = int(width_or_none * zoom_factor)
        mostrado = imutils.resize(bgr, width=target_width)
    else:
        h0, w0 = bgr.shape[:2]
        target_w = max(1, int(w0 * zoom_factor))
        mostrado = imutils.resize(bgr, width=target_w)
    h, w = mostrado.shape[:2]
    size_mostrada[label] = (w, h)
    rgb = cv2.cvtColor(mostrado, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=im, master=root)
    label.config(image=imgtk)
    label.image = imgtk

def actualizar_destino(nueva_img):
    global imagen_actual, roi_actual
    if usar_roi.get() and roi_actual is not None:
        roi_actual = asegurar_bgr(nueva_img)
        mostrar_imagen(roi_actual, lblROI, 300)
        set_status("Operación aplicada sobre ROI.")
    else:
        imagen_actual = asegurar_bgr(nueva_img)
        mostrar_imagen(imagen_actual, lblOutputImage)
        set_status("Operación aplicada sobre la imagen de trabajo.")

def obtener_fuente_operacion():
    global imagen_actual
    if imagen_actual is None: return None
    if usar_roi.get() and roi_actual is not None:
        return roi_actual.copy()
    return imagen_actual.copy()

# ==================
# Funciones de procesamiento (from funciones_principales.py)
# ==================

def histograma_grises():
    src = obtener_fuente_operacion()
    if src is None:
        set_status("No hay imagen de trabajo.")
        return
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    pixeles_totales = height * width
    valores = np.arange(256)
    cuenta = [0] * 256
    for fila in range(height):
        for columna in range(width):
            cuenta[gray[fila, columna]] += 1
    for i in range(len(cuenta)):
        cuenta[i] = cuenta[i] / pixeles_totales
    plt.figure()
    plt.title('Histograma de Grises')
    plt.xlabel('Nivel de Gris')
    plt.ylabel('Frecuencia Relativa')
    plt.bar(valores, cuenta, width=1, align='center', color='gray')
    plt.xlim([-1, 256])
    plt.show()
    set_status("Mostrando histograma de la imagen de trabajo.")

def ecualizacion():
    global imagen_operativa
    src = obtener_fuente_operacion()
    if src is None:
        set_status("No hay imagen de trabajo.")
        return

    # Si la imagen es color (3 canales)
    if len(src.shape) == 3 and src.shape[2] == 3:
        canales = cv2.split(src)
        canales_eq = [fc.ecualizacion(c) for c in canales]
        imagen_ecualizada = cv2.merge(canales_eq)
    else:
        imagen_ecualizada = fc.ecualizacion(src)

    imagen_operativa = imagen_ecualizada
    mostrar_imagen(imagen_operativa, lblOutputImage)



# ==================
# Funciones de kernel (from filtros.py)
# ==================
mascara_kernel = None
tipo_kernel_aplicar = None
entries = []
tamaño_var = None
sigma_var = None
tipo_var = None


def mascara():
    global imagen_operativa
    src = obtener_fuente_operacion()
    if src is None: 
        set_status("No hay imagen de trabajo.")
        return
        
    
    if len(src.shape) == 3:
        if tipo_kernel_aplicar in ["Prewitt Horizontal", "Prewitt Vertical","Sobel Horizontal","Sobel Vertival","Laplca"]:
            imagen = fc.mascara(src, mascara_kernel, tipo_kernel_aplicar, grises=False, estandarizar=True, prewitt=True)
        else:
            imagen = fc.mascara(src, mascara_kernel, tipo_kernel_aplicar, grises=False, estandarizar=True)
    else:
        if tipo_kernel_aplicar in ["Prewitt Horizontal", "Prewitt Vertical","Sobel Horizontal","Sobel Vertival","Laplace"]:
            imagen = fc.mascara(src, mascara_kernel, tipo_kernel_aplicar, grises=True, estandarizar=True, prewitt=True)
        else:
            imagen = fc.mascara(src, mascara_kernel, tipo_kernel_aplicar, grises=True, estandarizar=True)

    imagen_operativa = imagen
    mostrar_imagen(imagen_operativa, lblOutputImage)






# ==================
# Load and Restore
# ==================
def elegir_imagen():
    global imagen_original, imagen_actual, roi_actual, imagen_operativa
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
    if not file_path:
        return
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        set_status("No se pudo abrir la imagen.")
        return
    imagen_original = img.copy()
    imagen_actual = img.copy()
    roi_actual = None
    imagen_operativa = imagen_actual.copy()
    mostrar_imagen(imagen_original, lblInputImage, 300)
    mostrar_imagen(imagen_actual, lblOutputImage)
    mostrar_imagen(None, lblROI)
    usar_roi.set(False)
    set_status("Imagen cargada.")

def set_status(msg):
    lblStatus.config(text=msg)

def restaurar_original():
    global imagen_actual, roi_actual, imagen_operativa
    if imagen_original is None:
        return
    imagen_actual = imagen_original.copy()
    imagen_operativa = imagen_actual.copy()
    roi_actual = None
    mostrar_imagen(imagen_original, lblInputImage, 300)
    mostrar_imagen(imagen_actual, lblOutputImage)
    mostrar_imagen(None, lblROI)
    usar_roi.set(False)
    set_status("Se restauró la imagen original.")

# ==================
# Image Operations
# ==================
def aplicar_umbral_preview():
    global imagen_operativa
    src = obtener_fuente_operacion()
    if src is None:
        set_status("No hay imagen de trabajo.")
        return
        
    if len(src.shape) == 3:
        binaria = fc.funcion_umbral_preview(src, slider_umbral.get(), False, True)
    else:
        binaria = fc.funcion_umbral_preview(src, slider_umbral.get(), True, True)
    
    imagen_operativa = binaria
    mostrar_imagen(imagen_operativa, lblOutputImage)


def funcion_y_previw():
    global imagen_operativa
    src = obtener_fuente_operacion()
    if src is None: 
        set_status("No hay imagen de trabajo.")
        return
        
    if len(src.shape) == 3:
        imagen_gamma = fc.funcion_y_preview(src, slider_y.get(), False, True)
    else:
        imagen_gamma = fc.funcion_y_preview(src, slider_y.get(), True, True)
    
    imagen_operativa = imagen_gamma
    
    mostrar_imagen(imagen_operativa, lblOutputImage)


def fnegativo():
    global imagen_operativa
    src = obtener_fuente_operacion()
    if src is None: 
        set_status("No hay imagen de trabajo.")
        return
        
    bgr_src = asegurar_bgr(src)

    if len(bgr_src.shape) == 3:
        imagen_negativa = fc.fnegativo(bgr_src,False)
    else:
        imagen_negativa = fc.fnegativo(bgr_src,True)

    imagen_operativa = imagen_negativa
    mostrar_imagen(imagen_operativa, lblOutputImage)


 
media_gaussiana_valor = None
desvio_gaussiano_valor = None
rayleigh_valor = None
exponencial_valor = None
tipo_ruido = None
porcentaje_ruido = None
ruido_sal_pimineta = None

def generador_ruido(cantidad):
    global tipo_ruido, media_gaussiana_valor, desvio_gaussiano_valor, rayleigh_valor, exponencial_valor

    if tipo_ruido is None:
        set_status("Seleccione un tipo de ruido.")
        return None
    
    if tipo_ruido == "Gaussiano":
        return np.random.normal(media_gaussiana_valor, desvio_gaussiano_valor, cantidad)
    
    elif tipo_ruido == "Rayleigh":
        return np.random.rayleigh(rayleigh_valor, cantidad)
    elif tipo_ruido == "Exponencial":
        return np.random.exponential(exponencial_valor, cantidad)
    
    return None

def mostrar_histograma (datos_ruido,tipo_ruido_str):
    fig , ax = plt.subplots ( figsize =(6 , 4) )
    ax.hist (datos_ruido,bins =50 , density = True , alpha =0.6)
    ax.set_title ( f'Histograma del Ruido {tipo_ruido_str }')
    ax.set_xlabel ('Valor del Ruido ')
    ax.set_ylabel ('Densidad de Probabilidad ')
    ax.grid ( True )
    ventana_hist = tk.Toplevel ()
    ventana_hist.title ( f" Histograma de Ruido { tipo_ruido_str }"
    )
    canvas = FigureCanvasTkAgg(fig,master=ventana_hist)
    canvas.draw()
    canvas.get_tk_widget().pack( side = tk . TOP , fill = tk . BOTH ,
    expand =1)


def aplicar_ruido():
    global imagen_operativa, porcentaje_ruido, tipo_ruido, ruido
    if imagen_operativa is None:
        set_status("No hay imagen de trabajo.")
        return
    
    if len(imagen_operativa.shape) == 2:
        grises = imagen_operativa.copy()


        imagen_ruidosa = grises.copy().astype(np.float32)
        H, W = imagen_ruidosa.shape[:2]

        if tipo_ruido == "Sal y Pimienta":
            p = ruido_sal_pimineta
            for fila in range(H):
                for columna in range(W):
                    x = np.random.uniform(0, 1)
                    if x <= p:
                        imagen_ruidosa[fila, columna] = 0
                    elif x > 1 - p:
                        imagen_ruidosa[fila, columna] = 255
        else:
            cantidad_ruido = int(imagen_ruidosa.size * (porcentaje_ruido / 100))
            ruido = generador_ruido(cantidad_ruido)
            indices_pixels = np.random.choice(imagen_ruidosa.size, cantidad_ruido, replace=False)
            
            flat_img = imagen_ruidosa.ravel()

            if tipo_ruido == "Gaussiano":
                for i in range(len(ruido)):
                    flat_img[indices_pixels[i]] += ruido[i]
            elif tipo_ruido in ["Rayleigh", "Exponencial"]:
                for i in range (len(ruido)):
                    flat_img [indices_pixels[i]] *= ruido[i]

            imagen_ruidosa = ( (imagen_ruidosa - np.min(imagen_ruidosa)) / (np.max(imagen_ruidosa) - np.min(imagen_ruidosa)) ) * 255
    else:
        color = imagen_operativa.copy()

        imagen_ruidosa = color.copy().astype(np.float32)
        H, W, C = imagen_ruidosa.shape

        if tipo_ruido == "Sal y Pimienta":
            p = ruido_sal_pimineta
            for fila in range(H):
                for columna in range(W):
                    for c in range(C):
                        x = np.random.uniform(0, 1)
                        if x <= p:
                            imagen_ruidosa[fila, columna,c] = 0
                        elif x > 1 - p:
                            imagen_ruidosa[fila, columna,c] = 255
        else:
            cantidad_ruido = int(imagen_ruidosa.size * (porcentaje_ruido / 100))
            ruido = generador_ruido(cantidad_ruido)
            indices_pixels = np.random.choice(imagen_ruidosa.size, cantidad_ruido, replace=False)
            
            flat_img = imagen_ruidosa.ravel()

            if tipo_ruido == "Gaussiano":
                for i in range(len(ruido)):
                    flat_img[indices_pixels[i]] += ruido[i]
            elif tipo_ruido in ["Rayleigh", "Exponencial"]:
                for i in range (len(ruido)):
                    flat_img [indices_pixels[i]] *= ruido[i]

            imagen_ruidosa = ( (imagen_ruidosa - np.min(imagen_ruidosa)) / (np.max(imagen_ruidosa) - np.min(imagen_ruidosa)) ) * 255

    imagen_operativa = imagen_ruidosa.astype(np.uint8)
    mostrar_histograma(ruido, tipo_ruido)
    mostrar_imagen(imagen_operativa, lblOutputImage)

def pedir_parametros_ruido():
    global tipo_ruido, porcentaje_ruido
    global media_gaussiana_valor, desvio_gaussiano_valor
    global rayleigh_valor, exponencial_valor, ruido_sal_pimineta

    tipo_ruido = simpledialog.askstring("Tipo de ruido", "Ingrese tipo de ruido: Gaussiano, Rayleigh, Exponencial, Sal y Pimienta")
    if tipo_ruido is None:
        return

    if tipo_ruido == "Sal y Pimienta":
        ruido_sal_pimineta = simpledialog.askfloat("Probabilidad", "Ingrese p (0 - 0.5) para Sal y Pimienta:")
        if ruido_sal_pimineta is None: return
    else:
        porcentaje_ruido = simpledialog.askfloat("Porcentaje de ruido", "Ingrese porcentaje de ruido (0-100):")
        if porcentaje_ruido is None: return

    if tipo_ruido == "Gaussiano":
        media_gaussiana_valor = simpledialog.askfloat("Media", "Ingrese la media del ruido gaussiano:")
        if media_gaussiana_valor is None: return
        desvio_gaussiano_valor = simpledialog.askfloat("Desvío", "Ingrese el desvío del ruido gaussiano:")
        if desvio_gaussiano_valor is None: return
    elif tipo_ruido == "Rayleigh":
        rayleigh_valor = simpledialog.askfloat("Valor Rayleigh", "Ingrese el parámetro Rayleigh:")
        if rayleigh_valor is None: return
    elif tipo_ruido == "Exponencial":
        exponencial_valor = simpledialog.askfloat("Valor Exponencial", "Ingrese el parámetro Exponencial:")
        if exponencial_valor is None: return

    aplicar_ruido()


# Global variables
mascara_kernel = None
tipo_kernel_aplicar = None
entries = []
tamaño_var = None
sigma_var = None
tipo_var = None

def abrir_creador_kernel():
    global entries, tamaño_var, sigma_var, tipo_var, tipo_kernel_aplicar, mascara_kernel
    win = tk.Toplevel(root)
    win.title("Creador de Kernels")
    win.grab_set()
    win.configure(bg="#2c3e50")

    tk.Label(win, text="Tipo de Kernel", bg="#2c3e50", fg="#ecf0f1").grid(row=0, column=0, padx=5, pady=5)
    tipo_var = tk.StringVar(win)
    tipos = ["Media", "Mediana Ponderada", "Mediana", "Realce", "Gaussiano", "Personalizado", "Prewitt Horizontal", "Prewitt Vertical", "Sobel Horizontal", "Sobel Vertical","Laplace", "Marr Hildreth"]
    tipo_var.set(tipos[0])
    tk.OptionMenu(win, tipo_var, *tipos, command=lambda _: actualizar_grilla()).grid(row=0, column=1, padx=5, pady=5)

    tk.Label(win, text="Tamaño Kernel (impar)", bg="#2c3e50", fg="#ecf0f1").grid(row=1, column=0, padx=5, pady=5)
    tamaño_var = tk.IntVar(win)
    tamaño_var.set(3)
    tamaño_entry = tk.Entry(win, textvariable=tamaño_var, width=5)
    tamaño_entry.grid(row=1, column=1, padx=5, pady=5)
    tamaño_entry.bind('<KeyRelease>', lambda event: actualizar_grilla())

    tk.Label(win, text="Sigma (solo Gauss, Marr Hildreth)", bg="#2c3e50", fg="#ecf0f1").grid(row=2, column=0, padx=5, pady=5)
    sigma_var = tk.DoubleVar(win)
    sigma_var.set(1.0)
    tk.Entry(win, textvariable=sigma_var, width=5).grid(row=2, column=1, padx=5, pady=5)

        # Scrollable grid container
    canvas_frame = tk.Frame(win, bg="#2c3e50")
    canvas_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=10)

    canvas = tk.Canvas(canvas_frame, height=200, bg="#2c3e50", highlightthickness=0)
    scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # Grid container inside canvas
    frame_grid = tk.Frame(canvas, bg="#2c3e50")
    canvas.create_window((0, 0), window=frame_grid, anchor='nw')

    # Actualiza el área de scroll cuando el frame interno cambie de tamaño
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame_grid.bind("<Configure>", on_frame_configure)

    # (Opcional) Scroll con rueda del mouse
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def actualizar_grilla():
        global entries
        for w in frame_grid.winfo_children():
            w.destroy()

        tipo = tipo_var.get()

        try:
            if tipo == "Gaussiano":
                sigma = sigma_var.get()
                t = int((2 * sigma) + 1)  # Tamaño basado en sigma
                tamaño_var.set(t)

            elif tipo == "Marr Hildreth":
                sigma = sigma_var.get()
                t = int((4 * sigma) + 1)  # Tamaño más grande para LoG
                tamaño_var.set(t)
            else:
                t = tamaño_var.get()
                if t % 2 == 0 or t <= 0:
                    set_status("El tamaño del kernel debe ser un número impar y positivo.")
                    return
        except tk.TclError:
            set_status("Entrada de tamaño inválida. Debe ser un número entero.")
            return

        entries = [[None for _ in range(t)] for _ in range(t)]
        tipo = tipo_var.get()
        # Inicialización kernel
        if tipo == "Media":
            kernel_init = np.ones((t, t), dtype=np.float32)
        elif tipo == "Mediana Ponderada":
            if t == 3:
                kernel_init = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
            else:
                set_status("La mediana ponderada solo está implementada para 3x3.")
                kernel_init = np.zeros((t, t), dtype=np.float32)
        elif tipo == "Mediana":
            kernel_init = np.ones((t, t), dtype=np.float32)
        elif tipo == "Realce":
            kernel_init = np.full((t, t), -1, dtype=np.float32)
            centro = t // 2
            kernel_init[centro, centro] = (t * t) - 1
        elif tipo == "Prewitt Vertical":
            if t == 3:
                kernel_init = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            else:
                set_status("Prewitt solo está implementado para 3x3.")
                kernel_init = np.zeros((t, t), dtype=np.float32)
        elif tipo == "Prewitt Horizontal":
            if t == 3:
                kernel_init = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
            else:
                set_status("Prewitt solo está implementado para 3x3.")
                kernel_init = np.zeros((t, t), dtype=np.float32)

        elif tipo == "Sobel Vertical":
            if t == 3:
                kernel_init = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            else:
                set_status("Sobel solo está implementado para 3x3.")
                kernel_init = np.zeros((t, t), dtype=np.float32)
        elif tipo == "Sobel Horizontal":
            if t == 3:
                kernel_init = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
            else:
                set_status("Sobel solo está implementado para 3x3.")
                kernel_init = np.zeros((t, t), dtype=np.float32)
        elif tipo == "Laplace":
            if t == 3:
                kernel_init = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
            else:
                set_status("Laplace solo está implementado para 3x3.")
                kernel_init = np.zeros((t, t), dtype=np.float32)
        elif tipo == "Gaussiano":
            sigma = sigma_var.get()
            t = int((2*sigma )+1)
            ax = np.linspace(-(t - 1) / 2, (t - 1) / 2, t) # Genera un arreglo de coordenadas lineales entre -(t-1)/2 y (t-1)/2.
            xx, yy = np.meshgrid(ax, ax) # Crea dos matrices 2D que representan las coordenadas X e Y de una cuadrícula de tamaño t x t
            kernel_init = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        elif tipo == "Marr Hildreth":
            sigma = sigma_var.get()
            t = int((4 * sigma) + 1)
            ax = np.linspace(-(t - 1) / 2, (t - 1) / 2, t)
            xx, yy = np.meshgrid(ax, ax)
            r2 = xx**2 + yy**2
            kernel_init = ((r2 - 2 * sigma**2) / sigma**4) * np.exp(-r2 / (2 * sigma**2))


        elif tipo == "Personalizado":
            kernel_init = np.zeros((t, t), dtype=np.float32)

        for i in range(t):
            for j in range(t):
                entries[i][j] = tk.Entry(frame_grid, width=5)
                entries[i][j].grid(row=i, column=j, padx=1, pady=1)
                entries[i][j].insert(0, str(round(kernel_init[i, j], 3)))

    def aplicar_preview():
        global mascara_kernel, tipo_kernel_aplicar
        try:
            t = tamaño_var.get()
        except tk.TclError:
            set_status("Tamaño de kernel inválido.")
            return

        kernel_actual = np.zeros((t, t), dtype=np.float32)
        for i in range(t):
            for j in range(t):
                try:
                    kernel_actual[i, j] = float(entries[i][j].get())
                except ValueError:
                    set_status(f"Valor inválido en la celda ({i},{j}). Usando 0.")
                    kernel_actual[i, j] = 0
        mascara_kernel = kernel_actual
        tipo_kernel_aplicar = tipo_var.get()
        mascara()

    ttk.Button(win, text="Aplicar Kernel", command=aplicar_preview).grid(row=4, column=0, columnspan=2, pady=10)
    actualizar_grilla()

def resaltar_capa_preview(capa):
    global imagen_operativa
    src = obtener_fuente_operacion()
    if src is None: 
        set_status("No hay imagen de trabajo.")
        return
    bgr = asegurar_bgr(src)
    canal = {"B":0,"G":1,"R":2}[capa]
    salida = np.zeros_like(bgr)
    salida[:,:,canal] = bgr[:,:,canal]
    imagen_operativa = salida
    mostrar_imagen(imagen_operativa, lblOutputImage)

def aplicar_operacion_permanente():
    global imagen_actual, imagen_operativa
    if imagen_operativa is None:
        set_status("No hay una operación para aplicar.")
        return
    actualizar_destino(imagen_operativa)
    imagen_operativa = None



# ==================


def _coords_a_imagen(widget, x, y, base_img):
    if widget not in size_mostrada: return 0, 0
    disp_w, disp_h = size_mostrada[widget]
    H, W = base_img.shape[:2]
    fx = W / max(disp_w, 1)
    fy = H / max(disp_h, 1)
    xi = int(np.clip(x * fx, 0, W - 1))
    yi = int(np.clip(y * fy, 0, H - 1))
    return xi, yi

# ==================
# ROI and Auxiliary Functions
# ==================
sel_activa = False
x0, y0, x1, y1 = 0, 0, 0, 0
def comenzar_roi(event):
    global sel_activa, x0, y0
    if not editar_pixel.get():
        sel_activa = True
        x0, y0 = event.x, event.y
        set_status("Arrastra para seleccionar ROI...")

def arrastrar_roi(event):
    global x1, y1
    if sel_activa:
        x1, y1 = event.x, event.y

def finalizar_roi(event):
    global sel_activa, roi_actual
    if not sel_activa: return
    sel_activa = False

    widget = event.widget
    base = imagen_actual if widget is lblOutputImage else imagen_original
    if base is None: return

    xi0, yi0 = _coords_a_imagen(widget, x0, y0, base)
    xi1, yi1 = _coords_a_imagen(widget, x1, y1, base)

    x_min, x_max = sorted([xi0, xi1])
    y_min, y_max = sorted([yi0, yi1])

    if x_max - x_min < 2 or y_max - y_min < 2:
        set_status("ROI demasiado pequeña.")
        return

    roi = base[y_min:y_max, x_min:x_max].copy()
    roi_actual = asegurar_bgr(roi)
    prom_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean()
    prom_bgr = roi.mean(axis=(0,1)).astype(int)

    set_status(
        f"ROI capturada {x_min}:{x_max} x {y_min}:{y_max} | "
        f"Pixeles: {roi.shape[0]*roi.shape[1]} | "
        f"Prom. gris: {prom_gray:.1f} | "
        f"Prom. BGR: {prom_bgr.tolist()}"
    )
    mostrar_imagen(roi_actual, lblROI, 300)
    
def limpiar_roi():
    global roi_actual
    roi_actual = None
    mostrar_imagen(None, lblROI)
    usar_roi.set(False)
    set_status("ROI limpiada.")

def copiar_roi():
    global roi_actual
    if roi_actual is None:
        set_status("No hay ROI seleccionada.")
        return
    nueva = roi_actual.copy()
    actualizar_destino(nueva)
    set_status("ROI copiada como nueva imagen.")

# ==================
# UI
# ==================
# Top Frame: File and ROI actions
ttk.Label(frame_top, text="Operaciones de Archivo y ROI", font=("Helvetica", 12, "bold")).pack(pady=(0, 5))
frm_file_ops = ttk.Frame(frame_top)
frm_file_ops.pack(pady=5)

ttk.Button(frm_file_ops, text="Elegir imagen", command=elegir_imagen).pack(side="left", padx=5)
ttk.Button(frm_file_ops, text="Volver a ORIGINAL", command=restaurar_original).pack(side="left", padx=5)
ttk.Button(frm_file_ops, text="Guardar Resultado", command=lambda: guardar_imagen(imagen_actual)).pack(side="left", padx=5)
ttk.Button(frm_file_ops, text="Guardar ROI", command=lambda: guardar_imagen(roi_actual)).pack(side="left", padx=5)
ttk.Button(frm_file_ops, text="Copiar ROI como imagen", command=copiar_roi).pack(side="left", padx=5)
ttk.Button(frm_file_ops, text="Limpiar ROI", command=limpiar_roi).pack(side="left", padx=5)

# Image Display Frame
lbl_images_container = ttk.Frame(frame_images)
lbl_images_container.pack(fill="both", expand=True)


# Use grid for image labels for better alignment
lblInputImage = tk.Label(lbl_images_container, bd=2, relief="groove", bg="#1a252f")
lblInputImage.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
ttk.Label(lbl_images_container, text="Imagen Original").grid(row=1, column=0, pady=(0, 5))

lblOutputImage = tk.Label(lbl_images_container, bd=2, relief="groove", bg="#1a252f")
lblOutputImage.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
ttk.Label(lbl_images_container, text="Imagen Procesada").grid(row=1, column=1, pady=(0, 5))

lblROI = tk.Label(lbl_images_container, bd=2, relief="groove", bg="#1a252f")
lblROI.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
ttk.Label(lbl_images_container, text="Región de Interés (ROI)").grid(row=1, column=2, pady=(0, 5))

lbl_images_container.grid_columnconfigure(0, weight=1)
lbl_images_container.grid_columnconfigure(1, weight=1)
lbl_images_container.grid_columnconfigure(2, weight=1)
lbl_images_container.grid_rowconfigure(0, weight=1)


# Todas las funciones necesarias ya están incluidas localmente, no se requieren imports ni referencias a otros módulos.


def click_pixel(event):
    widget = event.widget
    if widget not in (lblOutputImage, lblInputImage, lblROI):
        return
    base = imagen_actual if widget is lblOutputImage else (imagen_original if widget is lblInputImage else roi_actual)
    if base is None:
        return
    xi, yi = _coords_a_imagen(widget, event.x, event.y, base)
    px = base[yi, xi]
    set_status(f"Pixel ({xi},{yi}) = {px.tolist()}")

# Bind mouse events
for lbl in (lblInputImage, lblOutputImage):
    lbl.bind("<Button-1>", comenzar_roi)
    lbl.bind("<B1-Motion>", arrastrar_roi)
    lbl.bind("<ButtonRelease-1>", finalizar_roi)
    lbl.bind("<Double-Button-1>", click_pixel)
lblROI.bind("<Double-Button-1>", click_pixel)

# Controls Frame: Processing options
ttk.Label(frame_controls, text="Controles de Procesamiento", font=("Helvetica", 12, "bold")).pack(pady=(0, 5))
frm_ops = ttk.Frame(frame_controls)
frm_ops.pack(pady=5)

# Dynamic controls section
dynamic_controls_frame = ttk.Frame(frm_ops)
dynamic_controls_frame.pack(side="left", padx=10)

# Dropdown for operations
opciones = ["Seleccionar Operación", "Umbral","Umbral iterativo","Umbral Otsu", "Gamma Correction", "Resaltar B", "Resaltar G", "Resaltar R", "Función Negativo", "Histograma grises", "Ecualización", "Ruido", "Filtro Kernel", "Prewitt Magnitud", "Sobel Magnitud", "Cruces por cero", "Cruces por umbral", "Difusion Isotropica","Difusion Anstropica", "Filtro Bilateral"]
opcion_seleccionada = tk.StringVar(root)
opcion_seleccionada.set(opciones[0])
menu_opciones = ttk.OptionMenu(frm_ops, opcion_seleccionada, *opciones, command=lambda op: mostrar_controles(op))
menu_opciones.pack(side="left", padx=5)





# --- Operaciones Mateamticas(Estandarización) ---
def sumar_otra_imagen():
    global imagen_actual, imagen_operativa
    if imagen_original is None:
        set_status("Primero cargá una imagen base.")
        return
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
    if not file_path: return
    otra = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if otra is None:
        set_status("No se pudo abrir la imagen a sumar.")
        return
    H, W = imagen_original.shape[:2]
    otra_rs = cv2.resize(otra, (W, H), interpolation=cv2.INTER_LINEAR)
    resultado = fc.sumar(imagen_original, otra_rs, estandarizar=True)
    imagen_actual = resultado
    mostrar_imagen(imagen_actual, lblOutputImage)
    set_status("Suma aplicada contra la ORIGINAL.")


def restar_otra_imagen():
    global imagen_actual, imagen_operativa, imagen_original
    if imagen_original is None:
        set_status("Primero cargá una imagen base.")
        return
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
    if not file_path: return
    otra = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if otra is None:
        set_status("No se pudo abrir la imagen a restar.")
        return
    H, W = imagen_original.shape[:2]
    otra_rs = cv2.resize(otra, (W, H), interpolation=cv2.INTER_LINEAR)
    resultado = fc.restar(imagen_original, otra_rs, estandarizar=True)
    imagen_actual = resultado
    mostrar_imagen(imagen_actual, lblOutputImage)
    set_status("Resta aplicada contra la ORIGINAL.")


def cuadrado_imagen():
    global imagen_actual, imagen_operativa
    if imagen_original is None:
        set_status("Primero cargá una imagen base.")
        return

    imagen_actual = fc.cuadrado_imagen(imagen_original, estandarizar=True)
    mostrar_imagen(imagen_actual, lblOutputImage)
    set_status("Potencia grado 2 aplicado a la original.")

def raiz_imagen():
    global imagen_actual, imagen_operativa
    if imagen_original is None:
        set_status("Primero cargá una imagen base.")
        return
    imagen_actual = fc.raiz_imagen(imagen_original, estandarizar=True)
    mostrar_imagen(imagen_actual, lblOutputImage)
    set_status("Raíz cuadrada aplicada a la ORIGINAL.")

def prewitt_magnitud():
    global imagen_actual, imagen_operativa
    if imagen_actual is None:
        set_status("Primero cargá una imagen base.")
        return
    imagen_original = imagen_actual.copy().astype(np.float32)

    mascara_horizontal = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    mascara_vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)

    if len(imagen_original.shape) == 2:

        prewitt_horizontal = fc.mascara(imagen_original, mascara=mascara_horizontal, tipo_kernel="Prewitt Horizontal",grises=True, prewitt=False)
        prewitt_vertical = fc.mascara(imagen_original, mascara=mascara_vertical, tipo_kernel="Prewitt Vertical",grises=True, prewitt=False)
    else:
        prewitt_horizontal = fc.mascara(imagen_original, mascara=mascara_horizontal, tipo_kernel="Prewitt Horizontal",grises=False, prewitt=False)
        prewitt_vertical = fc.mascara(imagen_original, mascara=mascara_vertical, tipo_kernel="Prewitt Vertical",grises=False, prewitt=False)
   
    magnitud = np.sqrt(prewitt_horizontal.astype(np.float32)**2 + prewitt_vertical.astype(np.float32)**2)
    magnitud = ((magnitud - np.min(magnitud)) / (np.max(magnitud) - np.min(magnitud))) * 255
    
    
    imagen_operativa = magnitud.astype(np.uint8)
    mostrar_imagen(imagen_operativa, lblOutputImage)
    set_status("Prewitt Magnitud aplicado a la imagen de trabajo.")

def sobel_magnitud():
    global imagen_actual, imagen_operativa
    if imagen_actual is None:
        set_status("Primero cargá una imagen base.")
        return
    imagen_original = imagen_actual.copy().astype(np.float32)

    mascara_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    mascara_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    if len(imagen_original.shape) == 2:
        sobel_horizontal = fc.mascara(imagen_original, mascara=mascara_horizontal, tipo_kernel="Sobel Horizontal",grises=True, prewitt=False)
        sobel_vertical = fc.mascara(imagen_original, mascara=mascara_vertical, tipo_kernel="Sobel Vertical",grises=True, prewitt=False)
    else:
        sobel_horizontal = fc.mascara(imagen_original, mascara=mascara_horizontal, tipo_kernel="Sobel Horizontal", prewitt=False)
        sobel_vertical = fc.mascara(imagen_original, mascara=mascara_vertical, tipo_kernel="Sobel Vertical", prewitt=False)




    magnitud = np.sqrt(sobel_horizontal.astype(np.float32)**2 + sobel_vertical.astype(np.float32)**2)
    magnitud = ((magnitud - np.min(magnitud)) / (np.max(magnitud) - np.min(magnitud))) * 255
    imagen_operativa = magnitud.astype(np.uint8)
    mostrar_imagen(imagen_operativa, lblOutputImage)
    set_status("Sobel Magnitud aplicado a la imagen de trabajo.")

umbral_cruces = None
laplace_gauss = None

def aplicar_cruces_simbolos():
    global imagen_actual, imagen_operativa, laplace_gauss
    if imagen_actual is None:
        set_status("Primero cargá una imagen base.")
        return

    sigma = laplace_gauss if laplace_gauss else 0.0

    imagen_original = imagen_actual.copy().astype(np.float32)

    if sigma > 0:
        t = int((4 * sigma) + 1)
        ax = np.linspace(-(t - 1) / 2, (t - 1) / 2, t)
        xx, yy = np.meshgrid(ax, ax)
        r2 = xx**2 + yy**2

        mascara = ((r2 - 2 * sigma**2) / sigma**4) * np.exp(-r2 / (2 * sigma**2))
        tipo = "Marr Hildreth"
        set_status(f"Aplicando Laplaciano del Gaussiano con sigma={sigma:.2f}")

    else:
        mascara = np.array([[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]], dtype=np.float32)
        tipo = "Laplace"
        set_status("Aplicando Laplaciano simple")

    if len(imagen_original.shape) == 3:
        laplace_img = fc.mascara(imagen_original, mascara, tipo, grises=False)
        imagen_actual = fc.aplicar_cruces(laplace_img, False)
    else:
        laplace_img = fc.mascara(imagen_original, mascara, tipo, grises=True)
        imagen_actual = fc.aplicar_cruces(laplace_img, True)


    imagen_operativa = imagen_actual
    mostrar_imagen(imagen_operativa, lblOutputImage)
    set_status(f"Cruces por signo aplicados ({tipo}).")


def aplicar_cruces_umbral():
    global umbral_cruces,imagen_actual, imagen_operativa,laplace_gauss
    if imagen_actual is None:
        set_status("Primero cargá una imagen base.")
        return

    sigma = laplace_gauss if laplace_gauss else 0.0

    imagen_original = imagen_actual.copy().astype(np.float32)

    if sigma > 0:
        t = int((4 * sigma) + 1)
        ax = np.linspace(-(t - 1) / 2, (t - 1) / 2, t)
        xx, yy = np.meshgrid(ax, ax)
        r2 = xx**2 + yy**2

        mascara = ((r2 - 2 * sigma**2) / sigma**4) * np.exp(-r2 / (2 * sigma**2))
        tipo = "Marr Hildreth"
        set_status(f"Aplicando Laplaciano del Gaussiano con sigma={sigma:.2f}")

    else:
        mascara = np.array([[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]], dtype=np.float32)
        tipo = "Laplace"
        set_status("Aplicando Laplaciano simple")

    if len(imagen_original.shape) == 3:
        laplace_img = fc.mascara(imagen_original, mascara, tipo, grises=False)
        imagen_actual = fc.aplicar_cruces_umbral(laplace_img,umbral_cruces, False)
    else:
        laplace_img = fc.mascara(imagen_original, mascara, tipo, grises=True)
        imagen_actual = fc.aplicar_cruces_umbral(laplace_img,umbral_cruces, True)

    imagen_operativa = imagen_actual
    mostrar_imagen(imagen_operativa, lblOutputImage)
    set_status("Cruces umbral aplicado a la imagen de trabajo.")


def pedir_umbral():
 global umbral_cruces,laplace_gauss

 umbral_cruces = simpledialog.askfloat("Umbral", "Ingrese el umbral para el cual se aplicara el cruce")
 laplace_gauss = simpledialog.askfloat("Sigma", "Poner 0 si Lapalce comun")
 aplicar_cruces_umbral()


def pedir_laplace():
 global laplace_gauss

 laplace_gauss = simpledialog.askfloat("Sigma", "Poner 0 si Lapalce comun")
 aplicar_cruces_simbolos()

t_difusion = None
lamba_anistropica = None

def difusion_isotropica():
    global t_difusion,imagen_actual, imagen_operativa,lamba_anistropica

    if imagen_actual is None:
        set_status("Primero cargá una imagen base.")
        return
    
    if len(imagen_original.shape) == 3:
        imagen_operativa = fc.anistropica(imagen_actual,t_anistropica= t_difusion,lamba_anistropica=lamba_anistropica,grises=False,isotropica=True)
    else:
        imagen_operativa = fc.anistropica(imagen_actual,t_anistropica=t_difusion,lamba_anistropica=lamba_anistropica,isotropica=True)

    mostrar_imagen(imagen_operativa, lblOutputImage)
    set_status("Difusion isotropica aplicado a la imagen de trabajo.")


t_anistropica = None
sigma_sensibilidad = None

def difusion_anistropica():
    global t_anistropica,imagen_actual, imagen_operativa,lamba_anistropica,sigma_sensibilidad

    if imagen_actual is None:
        set_status("Primero cargá una imagen base.")
        return
    
    if len(imagen_original.shape) == 3:
        imagen_operativa = fc.anistropica(imagen_actual,t_anistropica=t_anistropica,lamba_anistropica = lamba_anistropica, sigma_sensibilidad = sigma_sensibilidad,grises=False)
    else:
        imagen_operativa = fc.anistropica(imagen_actual,t_anistropica=t_anistropica,lamba_anistropica = lamba_anistropica,sigma_sensibilidad = sigma_sensibilidad)

    mostrar_imagen(imagen_operativa, lblOutputImage)
    set_status("Difusion anistropica aplicado a la imagen de trabajo.")


sigma_color = None
sigma_espacial = None

def filtro_bilateral():
    global imagen_actual, imagen_operativa,sigma_color,sigma_espacial

    if imagen_actual is None:
        set_status("Primero cargá una imagen base.")
        return
    

    t = int((2*sigma_espacial )+1)
    ax = np.linspace(-(t - 1) / 2, (t - 1) / 2, t) # Genera un arreglo de coordenadas lineales entre -(t-1)/2 y (t-1)/2.
    xx, yy = np.meshgrid(ax, ax) # Crea dos matrices 2D que representan las coordenadas X e Y de una cuadrícula de tamaño t x t
    mascara_gauss_espacial = np.exp(-(xx**2 + yy**2) / (2 * sigma_espacial**2))

    imagen_original = imagen_actual.copy().astype(np.float32)


    if len(imagen_original.shape) == 3:
        gauss_imagen = fc.mascara(imagen_original, mascara_gauss_espacial, "Gaussiano Color", grises=False,estandarizar=True,sigma_color = sigma_color)
    else:
        gauss_imagen = fc.mascara(imagen_original, mascara_gauss_espacial, "Gaussiano Color", grises=True,estandarizar=True,sigma_color = sigma_color)

    imagen_operativa = gauss_imagen


    mostrar_imagen(imagen_operativa, lblOutputImage)
    set_status("Filtro bilateral aplicado a la imagen de trabajo.")

t_inicial = None
t_predefinido = None

def umbralizacion_iterativa():
    global imagen_actual, imagen_operativa,t_inicial,t_predefinido

    if imagen_actual is None:
        set_status("Primero cargá una imagen base.")
        return
    
    imagen_original = imagen_actual.copy().astype(np.float32)


    if len(imagen_original.shape) == 3:

        umbral_it, umbral,iteraciones = fc.umbralizacion_iterativa(imagen_original,t_inicial,t_predefinido,grises=False)
    else:
        umbral_it, umbral,iteraciones = fc.umbralizacion_iterativa(imagen_original,t_inicial,t_predefinido,grises=True)

    imagen_operativa = umbral_it


    mostrar_imagen(imagen_operativa, lblOutputImage)
    set_status(f"Umbral iterativo aplicado a la imagen de trabajo. Umbral/es aplicados {umbral}. Con {iteraciones} iteraciones")


def umbralizacion_Otsu():
    global imagen_actual, imagen_operativa

    if imagen_actual is None:
        set_status("Primero cargá una imagen base.")
        return
    
    imagen_original = imagen_actual.copy().astype(np.float32)


    if len(imagen_original.shape) == 3:

        umbral_it, umbral = fc.umbralizacion_Otsu(imagen_original,grises=False)
    else:
        umbral_it, umbral = fc.umbralizacion_Otsu(imagen_original,grises=True)

    imagen_operativa = umbral_it


    mostrar_imagen(imagen_operativa, lblOutputImage)
    set_status(f"Umbral iterativo aplicado a la imagen de trabajo. Umbral/es aplicados {umbral}")

def pedir_umbral_iterativa():

    global t_inicial,t_predefinido

    t_inicial = simpledialog.askfloat("umbral inicial", "Ingrese umbral inicial")
    t_predefinido = simpledialog.askfloat("lamba", "Ingrese diferencia umbral")
    umbralizacion_iterativa()

def pedir_anistropica():
    global t_anistropica,lamba_anistropica,sigma_sensibilidad

    t_anistropica = simpledialog.askfloat("t", "Ingrese t iteraciones para la difusión")
    lamba_anistropica = simpledialog.askfloat("lamba", "Ingrese el lamba para la difusión")
    sigma_sensibilidad = simpledialog.askfloat("sigma", "Ingrese el sigma para la difusión")
    difusion_anistropica()


def pedir_t():
 global t_difusion,lamba_anistropica

 t_difusion = simpledialog.askfloat("t", "Ingrese el t para la difusión")
 lamba_anistropica = simpledialog.askfloat("lamba", "Ingrese el lamba para la difusión")

 difusion_isotropica()


def pedir_bilateral():
 global sigma_color,sigma_espacial

 sigma_color = simpledialog.askfloat("sigma color", "Ingrese el sigma color")
 sigma_espacial = simpledialog.askfloat("sigma espacial", "Ingrese el sigma espacial")

 filtro_bilateral()

# --- Math Operations Collapsible Panel ---
math_panel_container = ttk.Frame(frm_ops)
math_panel_container.pack(side="left", padx=10)

def toggle_math_panel():
    if math_panel.winfo_ismapped():
        math_panel.pack_forget()
    else:
        math_panel.pack(fill='x', padx=6, pady=4)

btn_toggle_math = ttk.Button(math_panel_container, text="Operaciones matemáticas ▾", command=toggle_math_panel)
btn_toggle_math.pack(side='top', fill='x', padx=6, pady=2)

math_panel = ttk.Frame(math_panel_container)

ttk.Button(math_panel, text="Sumar imagen", command=sumar_otra_imagen).pack(side='left', padx=8, pady=6)
ttk.Button(math_panel, text="Restar imagen", command=restar_otra_imagen).pack(side='left', padx=8, pady=6)
ttk.Button(math_panel, text="Cuadrado", command=cuadrado_imagen).pack(side='left', padx=8, pady=6)
ttk.Button(math_panel, text="Raíz cuadrada", command=raiz_imagen).pack(side='left', padx=8, pady=6)

# Other controls
ttk.Button(frm_ops, text="Aplicar", command=aplicar_operacion_permanente).pack(side="left", padx=5)
ttk.Checkbutton(frm_ops, text="Trabajar sobre ROI", variable=usar_roi).pack(side="left", padx=5)
editar_pixel = tk.BooleanVar(value=False)
ttk.Checkbutton(frm_ops, text="Editar Píxel", variable=editar_pixel).pack(side="left", padx=5)

# Global variables for sliders
slider_umbral = None
slider_y = None

def mostrar_controles(opcion):
    global slider_umbral, slider_y
    # Limpia el frame de controles dinámicos antes de añadir nuevos
    for widget in dynamic_controls_frame.winfo_children():
        widget.destroy()

    slider_umbral = None
    slider_y = None
    imagen_operativa = None
    mostrar_imagen(imagen_actual, lblOutputImage)

    if opcion == "Umbral":
        ttk.Label(dynamic_controls_frame, text="Umbral (gris)").pack(side="left")
        umbral_var = tk.IntVar(value=127)
        slider_umbral = ttk.Scale(dynamic_controls_frame, from_=0, to=255, orient="horizontal", variable=umbral_var, command=lambda val: aplicar_umbral_preview(), length=200)
        slider_umbral.pack(side="left", padx=5)
        umbral_entry = ttk.Entry(dynamic_controls_frame, textvariable=umbral_var, width=6)
        umbral_entry.pack(side="left")
        aplicar_umbral_preview()
    
    elif opcion == "Gamma Correction":
        ttk.Label(dynamic_controls_frame, text="Gamma (Y)").pack(side="left")
        gamma_var = tk.DoubleVar(value=1.0)
        
        # Se eliminó la opción 'resolution'
        slider_y = ttk.Scale(dynamic_controls_frame, from_=0.1, to=2.0,
                            orient="horizontal", variable=gamma_var,
                            command=lambda val: funcion_y_previw(), length=200, )
        slider_y.pack(side="left", padx=5)
        
        gamma_entry = ttk.Entry(dynamic_controls_frame, textvariable=gamma_var, width=6)
        gamma_entry.pack(side="left")
        funcion_y_previw()
    
    elif opcion.startswith("Resaltar"):
        capa = opcion.split(" ")[1]
        ttk.Button(dynamic_controls_frame, text="Previsualizar", command=lambda: resaltar_capa_preview(capa)).pack(side="left", padx=5)
    
    elif opcion == "Función Negativo":
        fnegativo()

    elif opcion == "Histograma grises":
        ttk.Button(dynamic_controls_frame, text="Mostrar Histograma", command=histograma_grises).pack(side="left", padx=5)
    
    elif opcion == "Ecualización":
        ttk.Button(dynamic_controls_frame, text="Aplicar Ecualización", command=ecualizacion).pack(side="left", padx=5)

    elif opcion == "Ruido":
        ttk.Button(dynamic_controls_frame, text="Configurar Ruido", command=pedir_parametros_ruido).pack(side="left", padx=5)

    elif opcion == "Filtro Kernel":
        ttk.Button(dynamic_controls_frame, text="Crear Kernel", command=abrir_creador_kernel).pack(side="left", padx=5)
    elif opcion == "Prewitt Magnitud":
        ttk.Button(dynamic_controls_frame, text="Aplicar Prewitt Magnitud", command=prewitt_magnitud).pack(side="left", padx=5)
    elif opcion == "Sobel Magnitud":
        ttk.Button(dynamic_controls_frame, text="Aplicar Sobel Magnitud", command=sobel_magnitud).pack(side="left", padx=5)
    elif opcion == "Cruces por cero":
        ttk.Button(dynamic_controls_frame, text="Aplicar Cruces por cero", command=pedir_laplace).pack(side="left", padx=5)
    elif opcion == "Cruces por umbral":
        ttk.Button(dynamic_controls_frame, text="Aplicar Cruces por umbral", command=pedir_umbral).pack(side="left", padx=5)
    elif opcion == "Difusion Isotropica":
        ttk.Button(dynamic_controls_frame, text="Aplicar Difusion Isotropica", command=pedir_t).pack(side="left", padx=5)
    elif opcion == "Difusion Anstropica":
        ttk.Button(dynamic_controls_frame, text="Aplicar Difusion Anstropica", command=pedir_anistropica).pack(side="left", padx=5)

    elif opcion == "Filtro Bilateral":
        ttk.Button(dynamic_controls_frame, text="Aplicar Filtro Bilateral", command=pedir_bilateral).pack(side="left", padx=5)

    elif opcion == "Umbral iterativo":
        ttk.Button(dynamic_controls_frame, text="Aplicar Umbral iterativo", command=pedir_umbral_iterativa).pack(side="left", padx=5)       

    elif opcion == "Umbral Otsu":
        ttk.Button(dynamic_controls_frame, text="Aplicar Umbral Otsu", command=umbralizacion_Otsu).pack(side="left", padx=5)



root.mainloop()