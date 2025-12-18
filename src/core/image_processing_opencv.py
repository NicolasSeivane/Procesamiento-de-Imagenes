"""
Procesamiento de Imágenes - Versión OpenCV Optimizada
=====================================================

Esta versión utiliza funciones nativas de OpenCV en lugar de loops manuales
para mejor rendimiento.

Author: Nicolas Seivane
Version: 2.0 (OpenCV Native)
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image 
import math
import numpy as np


def ecualizacion(grises):
    """
    Ecualización de histograma usando OpenCV nativo.
    Mucho más rápido que la implementación manual.
    """
    return cv2.equalizeHist(grises)


def restar(img1, img2, estandarizar=False):
    """Resta de imágenes usando operaciones vectorizadas de NumPy"""
    resultado = cv2.subtract(img1, img2)
    
    if estandarizar:
        resultado = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
    
    return resultado.astype(np.uint8)


def cuadrado_imagen(imagen, estandarizar=False):
    """Elevar al cuadrado usando operaciones vectorizadas"""
    imagen_cuadrada = np.square(imagen.astype(np.float32))
    
    if estandarizar:
        imagen_cuadrada = cv2.normalize(imagen_cuadrada, None, 0, 255, cv2.NORM_MINMAX)
    
    return imagen_cuadrada.astype(np.uint8)


def raiz_imagen(imagen, estandarizar=False):
    """Raíz cuadrada usando operaciones vectorizadas"""
    imagen_raiz = np.sqrt(imagen.astype(np.float32))
    
    if estandarizar:
        imagen_raiz = cv2.normalize(imagen_raiz, None, 0, 255, cv2.NORM_MINMAX)
    
    return imagen_raiz.astype(np.uint8)


def sumar(img1, img2, estandarizar=False):
    """Suma de imágenes usando OpenCV"""
    resultado = cv2.add(img1, img2)
    
    if estandarizar:
        resultado = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
    
    return resultado.astype(np.uint8)


def fnegativo(imagen, grises=False):
    """Negativo de imagen usando operación vectorizada"""
    return 255 - imagen


def funcion_y_preview(imagen, gamma, grises=True, estandarizar=False):
    """
    Corrección gamma usando LUT (Look-Up Table) de OpenCV.
    Mucho más eficiente que loops.
    """
    # Crear tabla de lookup
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    # Aplicar LUT
    imagen_gamma = cv2.LUT(imagen, table)
    
    if estandarizar:
        imagen_gamma = cv2.normalize(imagen_gamma, None, 0, 255, cv2.NORM_MINMAX)
    
    return imagen_gamma


def funcion_umbral_preview(imagen, umbral, grises=True, estandarizar=False, iterativo=False):
    """
    Umbralización usando cv2.threshold.
    Mucho más rápido que implementación manual.
    """
    if grises or not iterativo:
        _, imagen_umbral = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    else:
        # Para color con umbrales por canal
        if len(imagen.shape) == 3:
            canales = cv2.split(imagen)
            canales_umbral = []
            for i, canal in enumerate(canales):
                u = umbral[i] if isinstance(umbral, (list, tuple, np.ndarray)) else umbral
                _, canal_umbral = cv2.threshold(canal, u, 255, cv2.THRESH_BINARY)
                canales_umbral.append(canal_umbral)
            imagen_umbral = cv2.merge(canales_umbral)
        else:
            _, imagen_umbral = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    
    return imagen_umbral


def mascara(imagen, mascara_kernel, tipo_kernel="Media", grises=False, estandarizar=False, prewitt=False, sigma_color=None):
    """
    Aplicación de filtros usando cv2.filter2D.
    Optimizado con funciones nativas de OpenCV.
    """
    if tipo_kernel == "Media":
        # Usar cv2.blur para filtro de media
        ksize = mascara_kernel.shape[0]
        resultado = cv2.blur(imagen, (ksize, ksize))
    
    elif tipo_kernel == "Gaussiano":
        # Usar cv2.GaussianBlur
        ksize = mascara_kernel.shape[0]
        sigma = ksize / 6.0  # Aproximación estándar
        if sigma_color:
            # Filtro bilateral para Gaussiano con color
            resultado = cv2.bilateralFilter(imagen, ksize, sigma_color, sigma_color)
        else:
            resultado = cv2.GaussianBlur(imagen, (ksize, ksize), sigma)
    
    elif tipo_kernel == "Mediana":
        # Usar cv2.medianBlur
        ksize = mascara_kernel.shape[0]
        resultado = cv2.medianBlur(imagen, ksize)
    
    elif tipo_kernel in ["Sobel Horizontal", "Sobel Vertical"]:
        # Usar cv2.Sobel
        if tipo_kernel == "Sobel Horizontal":
            resultado = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
        else:
            resultado = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
        resultado = np.abs(resultado)
        if estandarizar or prewitt:
            resultado = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
        resultado = resultado.astype(np.uint8)
        return resultado
    
    elif tipo_kernel == "Laplace":
        # Usar cv2.Laplacian
        resultado = cv2.Laplacian(imagen, cv2.CV_64F)
        resultado = np.abs(resultado)
        if estandarizar or prewitt:
            resultado = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
        resultado = resultado.astype(np.uint8)
        return resultado
    
    else:
        # Para otros tipos, usar filter2D genérico
        resultado = cv2.filter2D(imagen, -1, mascara_kernel)
    
    if estandarizar and not prewitt:
        resultado = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
    
    if prewitt:
        resultado = np.abs(resultado)
        resultado = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
    
    return resultado.astype(np.uint8)


def umbralizacion_Otsu(imagen, grises=True):
    """
    Umbralización de Otsu usando cv2.threshold con THRESH_OTSU.
    Mucho más eficiente que implementación manual.
    """
    # Asegurar que la imagen es uint8
    if imagen.dtype != np.uint8:
        imagen = imagen.astype(np.uint8)
    
    if grises:
        umbral, imagen_binaria = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return imagen_binaria, int(umbral)
    else:
        # Para imágenes a color, aplicar a cada canal
        canales = cv2.split(imagen)
        umbrales = []
        canales_binarios = []
        
        for canal in canales:
            # Asegurar uint8
            if canal.dtype != np.uint8:
                canal = canal.astype(np.uint8)
            umbral, canal_bin = cv2.threshold(canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            umbrales.append(int(umbral))
            canales_binarios.append(canal_bin)
        
        imagen_binaria = cv2.merge(canales_binarios)
        return imagen_binaria, umbrales


def bordes_canny(magnitud, direccion, umbral1=100, umbral2=200):
    """
    Detección de bordes Canny usando cv2.Canny.
    Nota: Esta función ahora acepta una imagen directamente.
    """
    # Si se pasa magnitud y dirección, usar la imagen original
    if isinstance(magnitud, np.ndarray) and magnitud.dtype == np.uint8:
        return cv2.Canny(magnitud, umbral1, umbral2)
    else:
        # Convertir magnitud a uint8 si es necesario
        mag_uint8 = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.Canny(mag_uint8, umbral1, umbral2)


def anistropica(imagen_actual, t_anistropica, lamba_anistropica, sigma_sensibilidad=None, grises=True, estandarizar=True, isotropica=False):
    """
    Difusión anisotrópica.
    Nota: Esta función mantiene la implementación original ya que OpenCV
    no tiene una función nativa equivalente directa.
    """
    # Importar la función original
    from . import image_processing as ip_original
    return ip_original.anistropica(imagen_actual, t_anistropica, lamba_anistropica, 
                                   sigma_sensibilidad, grises, estandarizar, isotropica)


# Importar funciones que no tienen equivalente directo en OpenCV
# Estas mantienen la implementación original
from .image_processing import (
    aplicar_cruces,
    aplicar_cruces_umbral,
    susan_bordes,
    mostrar_tipo_susan,
    intercambio_de_pixeles,
    matriz_a_visual,
    transformada_de_hough,
    umbralizacion_iterativa,
    cambio_signo,
    Leclerc,
    Lorentz,
    funcion_g,
    derivadas
)


# Nota: Las funciones SIFT se mantienen en sift.py sin cambios
# ya que son implementaciones específicas del algoritmo
