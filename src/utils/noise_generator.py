import cv2
import numpy as np
import os

def agregar_ruido_gaussiano(imagen, media, sigma, porcentaje=1.0):
    imagen_float = imagen.astype(np.float32)
    ruido = np.random.normal(media, sigma, imagen.shape).astype(np.float32)
    # Crear máscara aleatoria para cada canal
    if imagen.ndim == 3:
        mascara = np.random.rand(*imagen.shape) < porcentaje  # (alto, ancho, canales)
    else:
        mascara = np.random.rand(*imagen.shape) < porcentaje  # (alto, ancho)
    imagen_float = np.where(mascara, imagen_float + ruido, imagen_float)
    imagen_ruidosa = np.clip(imagen_float, 0, 255).astype(np.uint8)
    return imagen_ruidosa


def procesar_carpeta(carpeta_entrada, carpeta_salida, media=0, sigma=25, sufijo="_ruidosa", porcentaje=1.0):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    for nombre_archivo in os.listdir(carpeta_entrada):
        ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
        if not os.path.isfile(ruta_entrada):
            continue
        imagen = cv2.imread(ruta_entrada)
        if imagen is None:
            print(f"No se pudo cargar {ruta_entrada}")
            continue
        imagen_ruidosa = agregar_ruido_gaussiano(imagen, media, sigma, porcentaje)
        nombre_salida = os.path.splitext(nombre_archivo)[0] + sufijo + os.path.splitext(nombre_archivo)[1]
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        cv2.imwrite(ruta_salida, imagen_ruidosa)
        print(f"Guardada: {ruta_salida}")

# Ejemplo de uso
if __name__ == "__main__":
    carpeta_entrada = "C:\\Users\\User\\Documents\\GitHub\\Procesamiento-de-Imagenes\\imagenesTP3\\imagenesApropiadasTP4"
    carpeta_salida = "C:\\Users\\User\\Documents\\GitHub\\Procesamiento-de-Imagenes\\imagenesTP3\\imagenes_contaminadas"
    # Solo el 30% de los píxeles tendrán ruido

    for porcentaje, sigma in [(0.1, 5), (0.3, 10), (0.5, 20)]:
            sufijo = f"_gauss_p{int(porcentaje*100)}_s{sigma}"
            procesar_carpeta(carpeta_entrada, carpeta_salida, media=0, sigma=sigma, sufijo=sufijo, porcentaje=porcentaje)