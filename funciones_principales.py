import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image 
import math
import numpy as np


def ecualizacion(grises):
    height, width = grises.shape
    imagen_ecualizada = np.zeros_like(grises)
    total = [0]*256
    for i in range(height):
        for j in range(width):
            total[grises[i,j]] += 1
    total_pixels = height * width
    cdf = [0]*256
    cdf[0] = total[0] / total_pixels
    for i in range(1, 256):
        cdf[i] = total[i]/total_pixels + cdf[i-1]
    min_index = 0
    for i in range(256):
        if cdf[i] > 0:
            min_index = i
            break
    min_val = cdf[min_index]
    for i in range(height):
        for j in range(width):
            val = grises[i,j]
            transformacion = ((cdf[val]-min_val)/(1-min_val))*255
            valor_final = transformacion
            imagen_ecualizada[i, j] = int(valor_final)
    return imagen_ecualizada

def restar(img1, img2, estandarizar=False):
    
    resultado = img1.astype(np.float32) - img2.astype(np.float32)

    if estandarizar:
        resultado = ( (resultado - np.min(resultado)) / (np.max(resultado) - np.min(resultado)) ) * 255
    
    resultado = resultado.astype(np.uint8)

    return resultado

def cuadrado_imagen(imagen, estandarizar=False):
    H, W = imagen.shape[:2]
    imagen_cuadrada = imagen.copy()
    for i in range(H):
        for j in range(W):
            imagen_cuadrada[i, j] = imagen[i, j]**2
    if estandarizar:
        imagen_cuadrada = ((imagen_cuadrada - np.min(imagen_cuadrada)) / (np.max(imagen_cuadrada) - np.min(imagen_cuadrada))) * 255
    imagen_cuadrada = imagen_cuadrada.astype(np.uint8)
    return imagen_cuadrada

def raiz_imagen(imagen, estandarizar=False):
    H, W = imagen.shape[:2]
    imagen_raiz = imagen.copy()
    for i in range(H):
        for j in range(W):
            imagen_raiz[i, j] = np.sqrt(imagen[i, j])
    if estandarizar:
        imagen_raiz = ((imagen_raiz - np.min(imagen_raiz)) / (np.max(imagen_raiz) - np.min(imagen_raiz))) * 255
    imagen_raiz = imagen_raiz.astype(np.uint8)
    return imagen_raiz


def sumar(img1, img2, estandarizar=False):

    resultado = img1.astype(np.int16) + img2.astype(np.int16)

    if estandarizar:
        resultado = ( (resultado - np.min(resultado)) / (np.max(resultado) - np.min(resultado)) ) * 255
    
    resultado = resultado.astype(np.uint8)

    return resultado

def fnegativo(imagen, grises=False):
    if grises is None:
        H, W, C = imagen.shape
        imagen_negativa = imagen.copy()
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    imagen_negativa[i, j, c] = 255 - imagen[i, j, c]
    else:
        H, W, C = imagen.shape
        imagen_negativa = imagen.copy()
        for i in range(H):
            for j in range(W):
                imagen_negativa[i, j] = 255 - imagen[i, j]
    imagen_negativa = imagen_negativa.astype(np.uint8)
    return imagen_negativa




def funcion_y_preview(imagen,gamma,grises=True,estandarizar=False):
    if grises:
        H, W = imagen.shape[:2]
        imagen_gamma = imagen.copy()
        constante = 255 **(1 - gamma)
        for i in range(H):
            for j in range(W):
                imagen_gamma[i, j] = 255 * (imagen[i, j] ** gamma) * constante
    else:
        H, W, C = imagen.shape
        imagen_gamma = imagen.copy()
        constante = 255 **(1 - gamma)
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    imagen_gamma[i, j, c] = (imagen[i, j, c] ** gamma) * constante
    if estandarizar:
        imagen_gamma = ((imagen_gamma - np.min(imagen_gamma)) / (np.max(imagen_gamma) - np.min(imagen_gamma))) * 255
    imagen_gamma = imagen_gamma.astype(np.uint8)
    return imagen_gamma


def funcion_umbral_preview(imagen,umbral,grises=True,estandarizar=False,iterativo = False):
    if grises:
        H, W = imagen.shape[:2]
        imagen_umbral = imagen.copy()
        for i in range(H):
            for j in range(W):
                if imagen[i, j] < umbral:
                    imagen_umbral[i, j] = 0
                else:
                    imagen_umbral[i, j] = 255
    elif iterativo is False and grises is False:
        H, W, C = imagen.shape
        imagen_umbral = imagen.copy()
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    if imagen[i, j, c] < umbral:
                        imagen_umbral[i, j, c] = 0
                    else:
                        imagen_umbral[i, j, c] = 255
    if iterativo is True and grises is False:
        H, W, C = imagen.shape
        imagen_umbral = imagen.copy()
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    if imagen[i, j, c] < umbral[c]:
                        imagen_umbral[i, j, c] = 0
                    else:
                        imagen_umbral[i, j, c] = 255



    if estandarizar:
        imagen_umbral = ((imagen_umbral - np.min(imagen_umbral)) / (np.max(imagen_umbral) - np.min(imagen_umbral))) * 255
    imagen_umbral = imagen_umbral.astype(np.uint8)
    return imagen_umbral

def aplicar_mascara(imagen, pixel, mascara, tipo,sigma_color=None):
    ancho = int((mascara.shape[0]// 2))
    nuevo_pixel = 0
    
    if tipo == "Media":
        for i in range(-ancho, ancho + 1):
            for j in range(-ancho, ancho + 1):
                x, y = pixel[0] + i, pixel[1] + j

                if 0 <= x < imagen.shape[0] and 0 <= y < imagen.shape[1]:

                    nuevo_pixel += imagen[x, y] * mascara[i + ancho, j + ancho]
        
        nuevo_pixel = nuevo_pixel/ (mascara.shape[0]*mascara.shape[1])

    elif tipo == "Gaussiano":
        for i in range(-ancho, ancho + 1):
            for j in range(-ancho, ancho + 1):
                x, y = pixel[0] + i, pixel[1] + j

                if 0 <= x < imagen.shape[0] and 0 <= y < imagen.shape[1]:

                    nuevo_pixel += imagen[x, y] * mascara[i + ancho, j + ancho]

        suma = mascara.sum()
        if suma != 0:
            nuevo_pixel = nuevo_pixel / suma
    
    elif tipo == "Gaussiano Color":
        suma_pesos = 0
        for i in range(-ancho, ancho + 1):
            for j in range(-ancho, ancho + 1):
                x, y = pixel[0] + i, pixel[1] + j

                if 0 <= x < imagen.shape[0] and 0 <= y < imagen.shape[1]:
                    valor_central = imagen[pixel[0], pixel[1]]
                    valor_vecino = imagen[x, y]

                    diff = valor_vecino - valor_central
                    peso_color = np.exp(-(diff**2) / (2 * sigma_color**2))

                    peso_total = mascara[i + ancho, j + ancho] * peso_color

                    nuevo_pixel += valor_vecino * peso_total
                    suma_pesos += peso_total

        if suma_pesos != 0:
            nuevo_pixel /= suma_pesos


    elif tipo == "Marr Hildreth":
        for i in range(-ancho, ancho + 1):
            for j in range(-ancho, ancho + 1):
                x, y = pixel[0] + i, pixel[1] + j

                if 0 <= x < imagen.shape[0] and 0 <= y < imagen.shape[1]:

                    nuevo_pixel += imagen[x, y] * mascara[i + ancho, j + ancho]


    elif tipo == "Realce":
        for i in range(-ancho, ancho + 1):
            for j in range(-ancho, ancho + 1):
                x, y = pixel[0] + i, pixel[1] + j

                if 0 <= x < imagen.shape[0] and 0 <= y < imagen.shape[1]:

                    nuevo_pixel += imagen[x, y] * mascara[i + ancho, j + ancho]


        nuevo_pixel = nuevo_pixel / (mascara.shape[0]*mascara.shape[1])

    elif tipo == "Mediana Ponderada":
            vecinos = []
            for i in range(-ancho, ancho + 1):
                for j in range(-ancho, ancho + 1):
                    x, y = pixel[0] + i, pixel[1] + j
                    if 0 <= x < imagen.shape[0] and 0 <= y < imagen.shape[1]:
                            
                            
                            for k in range(int((mascara[i + ancho, j + ancho]))):
                                
                                vecinos.append(imagen[x, y])
            
            vecinos.sort()
            nuevo_pixel = vecinos[len(vecinos)//2]

    elif tipo == "Mediana":
            vecinos = []
            for i in range(-ancho, ancho + 1):
                for j in range(-ancho, ancho + 1):
                    x, y = pixel[0] + i, pixel[1] + j
                    if 0 <= x < imagen.shape[0] and 0 <= y < imagen.shape[1]:
                                
                        vecinos.append(imagen[x, y])
            
            vecinos.sort()
            nuevo_pixel = vecinos[len(vecinos)//2]

    elif tipo in ["Prewitt Horizontal", "Prewitt Vertical","Sobel Horizontal", "Sobel Vertical","Laplace"]:
        for i in range(-ancho, ancho + 1):
            for j in range(-ancho, ancho + 1):
                x, y = pixel[0] + i, pixel[1] + j

                if 0 <= x < imagen.shape[0] and 0 <= y < imagen.shape[1]:

                    nuevo_pixel += imagen[x, y] * mascara[i + ancho, j + ancho]

    return nuevo_pixel


def mascara(imagen,mascara, tipo_kernel="Media", grises=False, estandarizar=False, prewitt=False,sigma_color=None):
    if grises:
        H, W = imagen.shape[:2]
        imagen_original = imagen.copy().astype(np.float32)
        imagen_mascara = np.zeros_like(imagen_original)
        for fila in range(H):
            for columna in range(W):
                imagen_mascara[fila, columna] = aplicar_mascara(imagen_original, [fila, columna], mascara, tipo_kernel,sigma_color)
    else:
        H, W, C = imagen.shape
        imagen_original = imagen.copy().astype(np.float32)
        imagen_mascara = np.zeros_like(imagen_original)
        for fila in range(H):
            for columna in range(W):
                for c in range(C):
                    imagen_mascara[fila, columna, c] = aplicar_mascara(imagen_original[:, :, c], [fila, columna], mascara, tipo_kernel,sigma_color)
   
   
    if estandarizar and not prewitt:
        imagen_mascara = ((imagen_mascara - np.min(imagen_mascara)) / (np.max(imagen_mascara) - np.min(imagen_mascara))) * 255
        imagen_mascara = imagen_mascara.astype(np.uint8)
    if prewitt:
        imagen_mascara = np.abs(imagen_mascara)
        print("Min:", np.min(imagen_mascara), "Max:", np.max(imagen_mascara))
        imagen_mascara = ( (imagen_mascara - np.min(imagen_mascara)) / (np.max(imagen_mascara) - np.min(imagen_mascara)) ) * 255
        imagen_mascara = imagen_mascara.astype(np.uint8)
    
    return imagen_mascara

def cambio_signo(a, b):
    return (a < 0 and b > 0) or (a > 0 and b < 0)


def aplicar_cruces(imagen, grises=True):

    if grises:
        H, W = imagen.shape
        imagen_cruces = np.zeros_like(imagen, dtype=np.uint8)

        for i in range(H):
            for j in range(W - 1):
                if cambio_signo(imagen[i, j], imagen[i, j + 1]):
                    imagen_cruces[i, j] = 255
                else:
                    imagen_cruces[i, j] = 0

        for j in range(W):
            for i in range(H - 1):
                if cambio_signo(imagen[i, j], imagen[i + 1, j]):
                    imagen_cruces[i, j] = 255

    else:
        H, W, C = imagen.shape
        imagen_cruces = np.zeros_like(imagen, dtype=np.uint8)

        for i in range(H):
            for j in range(W - 1):
                for c in range(C):
                    if cambio_signo(imagen[i, j, c], imagen[i, j + 1, c]):
                        imagen_cruces[i, j, c] = 255
                    else:
                        imagen_cruces[i, j, c] = 0

        for j in range(W):
            for i in range(H - 1):
                for c in range(C):
                    if cambio_signo(imagen[i, j, c], imagen[i + 1, j, c]):
                        imagen_cruces[i, j, c] = 255

    return imagen_cruces
                    



def aplicar_cruces_umbral(imagen, umbral, grises=True):
    if grises is True:
        H,W = imagen.shape[:2]
        imagen_cruces = np.zeros_like(imagen)
        for i in range(H - 1):
            for j in range(W - 1):
                if np.abs(imagen[i, j]+imagen[i, j + 1])> umbral or np.abs(imagen[i, j]+imagen[i+1, j]) > umbral:
                  imagen_cruces[i, j] = 255
                else:
                    imagen_cruces[i, j] = 0
    else:
        H,W, C = imagen.shape
        imagen_cruces = np.zeros_like(imagen)
        for i in range(H - 1):
            for j in range(W - 1):
                for c in range(C):
                    if np.abs(imagen[i, j, c]+imagen[i, j + 1, c]) > umbral or np.abs(imagen[i, j, c]+imagen[i+1, j, c]) > umbral:
                        imagen_cruces[i, j, c] = 255
                    else:
                        imagen_cruces[i, j, c] = 0



    imagen_cruces = imagen_cruces.astype(np.uint8)
    return imagen_cruces

def Leclerc(valor,sigma_sensibilidad=1):
    division = -(valor**2) / (sigma_sensibilidad**2)
    resultado = np.exp(division)
    return resultado

def Lorentz(valor,sigma_sensibilidad=1):
    division = -(valor**2) / (sigma_sensibilidad**2)
    resultado = 1 / (division + 1)
    return resultado

def funcion_g(valor,sigma_sensibilidad, funcion="Leclerc"):
    if funcion == "Leclerc":
        return Leclerc(valor,sigma_sensibilidad)
    elif funcion == "Lorentz":
        return Lorentz(valor,sigma_sensibilidad)
    else:
        raise ValueError("Función no válida. Usa 'Leclerc' o 'Lorentz'.")

def derivadas(imagen_actual,i,j,lamba_anistropica,sigma_sensibilidad=None,funcion="Leclerc",isotropica=False):
    if isotropica is False:   
        H, W = imagen_actual.shape[:2]
        suma = 0.0
        centro = imagen_actual[i, j]

        if j + 1 < W:  # Norte
            DN = imagen_actual[i, j + 1] - centro
            suma += DN * funcion_g(DN, sigma_sensibilidad, funcion)

        if i - 1 >= 0:  # Este
            DE = imagen_actual[i - 1, j] - centro
            suma += DE * funcion_g(DE, sigma_sensibilidad, funcion)

        if i + 1 < H:  # Oeste
            DO = imagen_actual[i + 1, j] - centro
            suma += DO * funcion_g(DO, sigma_sensibilidad, funcion)

        if j - 1 >= 0:  # Sur
            DS = imagen_actual[i, j - 1] - centro
            suma += DS * funcion_g(DS, sigma_sensibilidad, funcion)

        return lamba_anistropica * suma
    else:
        H, W = imagen_actual.shape[:2]
        suma = 0.0
        centro = imagen_actual[i, j]

        if j + 1 < W:  # Norte
            DN = imagen_actual[i, j + 1] - centro
            suma += DN 

        if i - 1 >= 0:  # Este
            DE = imagen_actual[i - 1, j] - centro
            suma += DE 

        if i + 1 < H:  # Oeste
            DO = imagen_actual[i + 1, j] - centro
            suma += DO 

        if j - 1 >= 0:  # Sur
            DS = imagen_actual[i, j - 1] - centro
            suma += DS 

        return lamba_anistropica * suma

                    
def anistropica(imagen_actual,t_anistropica,lamba_anistropica,sigma_sensibilidad=None,grises=True,estandarizar=True,isotropica=False):
    if isotropica is False:
        if grises:
            H, W = imagen_actual.shape[:2]
            imagen_asintropica = imagen_actual.copy().astype(np.float32)

            for t in range(int(t_anistropica)):


                for i in range(H):
                    for j in range(W):                 
                            imagen_asintropica[i, j] += derivadas(imagen_asintropica,i, j,sigma_sensibilidad=sigma_sensibilidad,lamba_anistropica=lamba_anistropica)

                        
        else:
            H, W, C = imagen_actual.shape
            imagen_asintropica = imagen_actual.copy().astype(np.float32)

            for t in range(int(t_anistropica)):

                for i in range(H):
                    for j in range(W):
                        for c in range(C):
                                imagen_asintropica[i, j, c] += derivadas(imagen_asintropica[:, :, c], i, j,sigma_sensibilidad=sigma_sensibilidad,lamba_anistropica=lamba_anistropica)
                            
        if estandarizar:
            imagen_asintropica = ((imagen_asintropica - np.min(imagen_asintropica)) / (np.max(imagen_asintropica) - np.min(imagen_asintropica))) * 255
        imagen_asintropica = imagen_asintropica.astype(np.uint8)

        return imagen_asintropica
    
    else:
        if grises:
            H, W = imagen_actual.shape[:2]
            imagen_asintropica = imagen_actual.copy().astype(np.float32)

            for t in range(int(t_anistropica)):


                for i in range(H):
                    for j in range(W):
                            imagen_asintropica[i, j] += derivadas(imagen_asintropica,i, j,lamba_anistropica=lamba_anistropica,isotropica=True)

                        
        else:
            H, W, C = imagen_actual.shape
            imagen_asintropica = imagen_actual.copy().astype(np.float32)

            for t in range(int(t_anistropica)):

                for i in range(H):
                    for j in range(W):
                        for c in range(C):
                                imagen_asintropica[i, j, c] += derivadas(imagen_asintropica[:, :, c], i, j,lamba_anistropica=lamba_anistropica,isotropica=True)
                            
        if estandarizar:
            imagen_asintropica = ((imagen_asintropica - np.min(imagen_asintropica)) / (np.max(imagen_asintropica) - np.min(imagen_asintropica))) * 255
        imagen_asintropica = imagen_asintropica.astype(np.uint8)

        return imagen_asintropica


def umbralizacion_iterativa(imagen_original, t_inicial, t_predefinido, grises=False):
    if grises:
        # Umbralización en escala de grises
        t = t_inicial
        iteracion = 0
        while True:
            iteracion += 1
            G1 = imagen_original[imagen_original > t]
            G2 = imagen_original[imagen_original <= t]

            if G1.size == 0 or G2.size == 0:
                break

            M1, M2 = G1.mean(), G2.mean()
            t_nuevo = (M1 + M2) / 2

            if abs(t_nuevo - t) < t_predefinido:
                t = int(round(t_nuevo))
                break
            t = t_nuevo

        imagen_binaria = funcion_umbral_preview(imagen_original, t, grises=True,estandarizar=True)
        return imagen_binaria, t,iteraciones

    else:
        canales = cv2.split(imagen_original)
        umbrales = []
        iteraciones = []

        for canal in canales:
            t = t_inicial
            iteracion = 0
            while True:
                iteracion += 1
                G1 = canal[canal > t]
                G2 = canal[canal <= t]

                if G1.size == 0 or G2.size == 0:
                    break

                M1, M2 = G1.mean(), G2.mean()
                t_nuevo = (M1 + M2) / 2

                if abs(t_nuevo - t) < t_predefinido:
                    t = int(round(t_nuevo))
                    break
                t = t_nuevo

            umbrales.append(t)
            iteraciones.append(iteracion)

        
        imagen_binaria = funcion_umbral_preview(imagen_original, umbrales, grises=False,estandarizar=True,iterativo=True)
        return imagen_binaria, umbrales,iteraciones
    
def umbralizacion_Otsu(imagen,grises=True):
        
    if grises is True:    
        height, width = imagen.shape[:2]
        
        total_frec = [0]*256
        for i in range(height):
            for j in range(width):
                val = int(imagen[i,j])
                total_frec[val] += 1

        total_pixels = height * width

        # Probabilidades 
        pi = [f/total_pixels for f in total_frec]

        # Probabilidad acumulada
        Pi = [0]*256
        for i in range(256):
            Pi[i] = sum(pi[:i])

        mt = [0]*256

        for i in range(256):
            mt[i] = sum([pi[v]*v for v in range(i)])

        mg = mt[-1]

        o_bt = [0]*256

        for i in range(256):
            if Pi[i] == 0 or Pi[i] == 1:
                o_bt[i] = 0
            else:
                o_bt[i] = ((mg*Pi[i] - mt[i])**2) / (Pi[i]*(1-Pi[i]))

        t = np.argmax(o_bt)
        t = int(round(t))

        imagen_binaria = funcion_umbral_preview(imagen, t, grises=True,estandarizar=True)
        return imagen_binaria, t

    else:

        canales = cv2.split(imagen)
        umbrales = []

        for canal in canales:
            height, width = imagen.shape[:2]
        
            total_frec = [0]*256
            for i in range(height):
                for j in range(width):
                    val = int(canal[i,j])
                    total_frec[val] += 1

            total_pixels = height * width

            # Probabilidades 
            pi = [f/total_pixels for f in total_frec]

            # Probabilidad acumulada
            Pi = [0]*256
            for i in range(256):
                Pi[i] = sum(pi[:i])

            mt = [0]*256

            for i in range(256):
                mt[i] = sum([pi[v]*v for v in range(i)])

            mg = mt[-1]

            o_bt = [0]*256

            for i in range(256):
                if Pi[i] == 0 or Pi[i] == 1:
                    o_bt[i] = 0
                else:
                    o_bt[i] = ((mg*Pi[i] - mt[i])**2) / (Pi[i]*(1-Pi[i]))

            t = np.argmax(o_bt)
            t = int(round(t))
            umbrales.append(t)

        imagen_binaria = funcion_umbral_preview(imagen, umbrales, grises=False,estandarizar=True,iterativo=True)

        return imagen_binaria, umbrales



    
def bordes_canny(magnitud,direccion, umbral1=100, umbral2=200):

    h, w = magnitud.shape[:2]
    Z = np.zeros((h,w), dtype=np.float32)
    sectors = np.zeros_like(direccion, dtype=np.uint8)
    sectors[((direccion > -22.5) & (direccion <= 22.5)) |
        ((direccion <= -157.5) | (direccion > 157.5))] = 0

    sectors[((direccion > 22.5) & (direccion <= 67.5)) |
            ((direccion <= -112.5) & (direccion > -157.5))] = 45

    sectors[((direccion > 67.5) & (direccion <= 112.5)) |
            ((direccion <= -67.5) & (direccion > -112.5))] = 90

    sectors[((direccion > 112.5) & (direccion <= 157.5)) |
            ((direccion <= -22.5) & (direccion > -67.5))] = 135

    for i in range(1, h-1):
        for j in range(1, w-1):
            val = magnitud[i, j]
            sector = sectors[i, j]

            if sector == 0:      # Horizontal
                if val >= magnitud[i, j+1] and val >= magnitud[i, j-1]:
                    Z[i, j] = val
            elif sector == 45:   # Diagonal ↗
                if val >= magnitud[i-1, j+1] and val >= magnitud[i+1, j-1]:
                    Z[i, j] = val
            elif sector == 90:   # Vertical
                if val >= magnitud[i-1, j] and val >= magnitud[i+1, j]:
                    Z[i, j] = val
            else:                # 135° Diagonal ↖
                if val >= magnitud[i-1, j-1] and val >= magnitud[i+1, j+1]:
                    Z[i, j] = val

    Z[Z < umbral1] = 0
    Z[Z >= umbral2] = 255
    mid = (Z >= umbral1) & (Z < umbral2)
    Z[mid] = 128

    Z_copy = Z.copy()
    cambios = True
    while cambios:
        cambios = False
        for i in range(1, h-1):
            for j in range(1, w-1):
                if Z_copy[i,j] == 128:
                    vecinos = [Z_copy[i-1,j], Z_copy[i+1,j], Z_copy[i,j-1], Z_copy[i,j+1],
                            Z_copy[i-1,j-1], Z_copy[i-1,j+1], Z_copy[i+1,j-1], Z_copy[i+1,j+1]]
                    if any(v == 255 for v in vecinos):
                        Z_copy[i,j] = 255
                        cambios = True
                    else:
                        Z_copy[i,j] = 0
    Z = Z_copy
    bordes = Z.astype(np.uint8)

    return bordes


def susan_bordes(imagen,umbral=15, tipo="borde"):
            H, W = imagen.shape
            imagen_susan = np.zeros_like(imagen, dtype=np.uint8)

            for i in range(H):
                for j in range(W):
                    centro = imagen[i, j]
                    suma = 0

                    for di in range(-3, 4):
                        for dj in range(-3, 4):
                            if di*di + dj*dj <= 9:
                                x, y = i + di, j + dj

                                if 0 <= x < H and 0 <= y < W:
                                    diff = abs(imagen[x, y] - centro)
                                    if diff < umbral:
                                        suma += 1
                    sr = 1 - suma/37
                    if sr < 0.35:
                        imagen_susan[i, j] = 0
                    elif sr > 0.35 and sr < 0.65:
                        imagen_susan[i, j] = 1
                    elif sr >= 0.65:
                        imagen_susan[i, j] = 2

            imagen_susan = mostrar_tipo_susan(imagen_susan, imagen, tipo)

            return imagen_susan

def mostrar_tipo_susan(imagen_susan, imagen_original, tipo):
        imagen_bgr = cv2.cvtColor(imagen_original, cv2.COLOR_GRAY2BGR)
         
        # Crear máscara según tipo
        if tipo == "borde":
            mascara = imagen_susan == 1
        elif tipo == "esquina":
            mascara = imagen_susan == 2
        elif tipo == "ambos":
            mascara = (imagen_susan == 1) | (imagen_susan == 2)

        # Pintamos sobre la imagen original en verde
        imagen_bgr[mascara] = [0, 0, 255]  # rojo sólido


        return imagen_bgr.astype(np.uint8)

def intercambio_de_pixeles(imagen, promedios, matriz):

    theta0, theta1 = promedios

    h, w = imagen.shape
    matriz = matriz.copy()
    
    # Distancia Fd(x)
    dist0 = np.abs(imagen - theta0) + 1e-6
    dist1 = np.abs(imagen - theta1) + 1e-6
    Fd = np.log(dist0 / dist1)

    # --- Paso 1: Lout -> Lin ---
    Lout_coords = np.argwhere(matriz == 1)
    for x, y in Lout_coords:
        if Fd[x, y] > 0:
            matriz[x, y] = -1  # mover a Lin
            # vecinos 4-conectados
            for i, j in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if 0 <= i < h and 0 <= j < w:
                    if matriz[i,j] == 3:
                        matriz[i,j] = 1  # agregar a Lout

    # --- Paso 2: Actualizar Lin interior ---
    Lin_coords = np.argwhere(matriz == -1)
    for x, y in Lin_coords:
        # si el píxel ya es interior según vecinos (ejemplo simple)
        # se convierte en objeto
        vecinos = []
        for i, j in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
            if 0 <= i < h and 0 <= j < w:
                vecinos.append(matriz[i,j])
        if all(v != 3 and v != 1 for v in vecinos):
            matriz[x, y] = -3  # ahora interior

    # --- Paso 3: Lin -> Lout ---
    Lin_coords = np.argwhere(matriz == -1)
    for x, y in Lin_coords:
        if Fd[x, y] < 0:
            matriz[x, y] = 1  # mover a Lout
            # vecinos 4-conectados
            for i, j in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if 0 <= i < h and 0 <= j < w:
                    if matriz[i,j] == -3:
                        matriz[i,j] = -1  # agregar a Lin

    # --- Paso 4: Actualizar Lout exterior ---
    Lout_coords = np.argwhere(matriz == 1)
    for x, y in Lout_coords:
        vecinos = []
        for i, j in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
            if 0 <= i < h and 0 <= j < w:
                vecinos.append(matriz[i,j])
        if all(v != -3 and v != -1 for v in vecinos):
            matriz[x, y] = 3  # ahora exterior

    return matriz

def matriz_a_visual(imagen, matriz, alpha=0.5):

    # Aseguramos que la imagen sea RGB
    if len(imagen.shape) == 2:
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        imagen_rgb = imagen.copy()

    # Creamos la máscara en color
    mask = np.zeros_like(imagen_rgb, dtype=np.uint8)

    # Lout (azul)
    mask[matriz == 1] = [255, 0, 0]

    # Lin (magenta)
    mask[matriz == -1] = [255, 0, 255]

    # Fondo no se pinta, queda transparente (la máscara sigue en negro, no cambia la imagen original)

    # Superposición con transparencia
    visual = cv2.addWeighted(imagen_rgb, 1 - alpha, mask, alpha, 0)

    return visual

def transformada_de_hough(imagen_actual, imagen_original, umbral, epsilon = 0.5):

    alto, ancho = imagen_actual.shape[:2]

    D = max(alto, ancho)
    r_max = np.sqrt(2) * D
    N_theta = 180
    N_r = int(2 * r_max)

    theta = np.deg2rad(np.linspace(-90, 90, N_theta, endpoint=False))
    r = np.linspace(-r_max, r_max, N_r)

    matriz_parametros = np.zeros((N_r, N_theta), dtype=np.float32)

    y_blancos, x_blancos = np.nonzero(imagen_actual)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
# Calcular r para cada píxel blanco y cada ángulo 
# for x, y in zip(x_blancos, y_blancos):
#  r_values = x * cos_theta + y * sin_theta
#  r_indices = np.round((r_values + r_max) * (N_r / (2 * r_max))).astype(int)
#  ''' Primero los paso a positivos, porque pueden ser negativos en: (r_values + r_max)
#  Luego se calcula el factor de la escala para mapear al rango discreto [0 , N_r -1] 
# Luego se aplica el factor de la escala para convertirlos en [0,N_r-1] Luego se redondea '''
#  # Incrementar los valores en la matriz de parámetros 
#  valid_indices = (r_indices >= 0) & (r_indices < N_r) 
#  matriz_parametros[r_indices[valid_indices], np.arange(N_theta)[valid_indices]] += 1
    # Calcular r para cada píxel blanco y cada ángulo
    for x, y in zip(x_blancos, y_blancos):
        for j, theta_j in enumerate(theta):
            r_val = x * np.cos(theta_j) + y * np.sin(theta_j)
            # buscar todos los r_i que cumplen la condición
            valid_r = np.where(np.abs(r - r_val) < epsilon)[0] # en unidades de r
            matriz_parametros[valid_r, j] += 1

    # Dibujar las líneas detectadas
    imagen_color = cv2.cvtColor(imagen_original, cv2.COLOR_GRAY2BGR)
    indices = np.argwhere(matriz_parametros > umbral)

    print(f'Se van a dibujar {len(indices)} rectas')
    for r_i, i_theta in indices:
        rho = r[r_i]
        t = theta[i_theta]
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * rho
        y0 = b * rho

        # Puntos extremos de la recta
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Validar coordenadas antes de dibujar
        cv2.line(imagen_color, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return imagen_color

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
###  SIFT =========================================================================================================================================================================
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 



def symmetrizationZ2 (k, M):
    return min([k%(2*M), (2*M-1-k)%(2*M)])
    
# Function to compute the bilinear interpolation of an image

def bilinear_interpolation(image, 
                           delta = 0.5 # inter-pixel distance of the output image
                           ):
    M = image.shape[0]
    N = image.shape[1]
    
    Mp = math.floor(M/delta)
    Np = math.floor(N/delta)

    u = np.zeros((Mp, Np))
    for m in range(Mp):
        for n in range(Np):
            x = m*delta
            y = n*delta
            xf = math.floor(x)
            yf = math.floor(y)
            u[m,n] = ((x-xf)*(y-yf)*image[symmetrizationZ2(xf+1,M), symmetrizationZ2(yf+1,N)] + 
                (1+xf-x)*(y-yf)*image[symmetrizationZ2(xf,M), symmetrizationZ2(yf+1,N)] +
                (x-xf)*(1+yf-y)*image[symmetrizationZ2(xf+1,M), symmetrizationZ2(yf,N)] +
                (1+xf-x)*(1+yf-y)*image[symmetrizationZ2(xf,M), symmetrizationZ2(yf,N)])
            
    return u

# Function to compute the Gaussian smoothing

def gaussian_smoothing(sigma):
    
    bound = math.ceil(4*sigma)
    kp = np.arange(-bound, bound + 1)
    
    g = np.exp(-kp**2 / (2 * sigma**2))
    g /= g.sum()
    
    return kp, g

    
# Function to apply the Gaussian convolution

def gaussian_convolution(image, sigma):
    
    M, N = image.shape
    indices, kernel = gaussian_smoothing(sigma)
    
    Gu = np.zeros((M, N))
    
    for m in range(M):
        sym_rows = np.array([symmetrizationZ2(m+i, M) for i in indices])
        for n in range(N):             
            sym_cols = np.array([symmetrizationZ2(n+j, N) for j in indices])
            window = image[np.ix_(sym_rows, sym_cols)]
            Gu[m, n] = np.matmul(kernel[None, :], np.matmul(window, kernel[:, None])).item()
                
    return Gu


# Function to compute the digital Gaussian scale-space

def digital_Gaussian_scale_space(image, 
                                 delta = 0.5,
                                 number_octaves = 4,
                                 number_scales = 3, # per octave
                                 sigma_min = 0.8, # blur level in the seed image
                                 delta_min = 0.5, # inter-sample distance in the seed image
                                 sigma_in = 0.5 # asumed blur level in the input image
                                 ):
    
    # Initialize the octaves
    octaves = []
    
    # Compute the first octave
    
    octaves1 = []
    
    # Interpolate the original image
    u = bilinear_interpolation(image, delta_min)
    
    # Gaussian blur
    sigma0 = np.sqrt(sigma_min**2 - sigma_in**2)/delta_min
    v = gaussian_convolution(u, sigma = sigma0)
    octaves1.append(v)
    
    # Compute the other images in the first octave
    for s in range(1,number_scales+3):
        print(s)
        rho = sigma_min/delta_min * np.sqrt(2**(2*s/number_scales)-2**(2*(s-1)/number_scales))
        v = gaussian_convolution(v, sigma = rho)
        octaves1.append(v)
    
    octaves.append(octaves1)
    # Compute the subsequent octaves
    
    M, N = image.shape
    
    for o in range(1,number_octaves):
        
        octaves1 = []
      
        M0 = math.floor(2**(1-o) * M)
        N0 = math.floor(2**(1-o) * N)
        
        # Compute the first image in the octave by subsampling
        v = np.zeros((M0, N0))
        for m in range(M0):
            for n in range(N0):
                v[m,n] = octaves[o-1][number_scales][2*m,2*n]
        octaves1.append(v)
                
        # Compute the other images in the octave
        for s in range(1,number_scales+3):
            print(s)
            rho = sigma_min/delta_min * np.sqrt(2**(2*s/number_scales)-2**(2*(s-1)/number_scales))
            v = gaussian_convolution(v, sigma = rho)
            octaves1.append(v)
            
        octaves.append(octaves1)
        
    return(octaves)


# Function to compute the difference of Gaussian scale-space

def DoG (set_of_octaves):
    
    dog = []
    
    for o in range(len(set_of_octaves)):
        
        dog_inner = []
        
        for s in range(len(set_of_octaves[0])-1):
            dog_inner.append(set_of_octaves[o][s+1] - set_of_octaves[o][s])
            
        dog.append(dog_inner)
        
    return(dog)

# Function to scann for the 3D discrete extrema of the DoG

def extrema_DoG (dog):
    
    extrema = []
    
    for o in range(len(dog)):
        M, N = dog[o][0].shape
        
        for s in range(1,len(dog[0])-1): 
            
            for m in range(1, M-1):
                for n in range(1, N-1):
                    px = dog[o][s][m,n]
                    neighbors = np.concatenate(
                        (dog[o][s-1][range(m-1,m+2),:][:,range(n-1,n+2)].flatten(),
                         np.delete(dog[o][s][range(m-1,m+2),:][:,range(n-1,n+2)].flatten(),4),
                         dog[o][s+1][range(m-1,m+2),:][:,range(n-1,n+2)].flatten()))
                
                    if (px > np.max(neighbors)) or (px < np.min(neighbors)):
                        extrema.append([o,s,m,n])
    
    return extrema

# Function to discard low contrasted candidate keypoints

def filter_extrema_DoG (dog, extrema, threshold_dog = 0.015):
    
    kept_extrema = []
    
    for i in range(len(extrema)):
        if dog[extrema[i][0]][extrema[i][1]][extrema[i][2], extrema[i][3]] >= 0.8 * threshold_dog:
            kept_extrema.append(extrema[i])
            
    return kept_extrema
    
# Function to compute the quadratic interpolation on a discrete DoG sample

def quadratic_interpolation (dog, sample):
    
    current_octave = dog[sample[0]]
    previous_scale = current_octave[sample[1]-1][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
    current_scale = current_octave[sample[1]][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
    posterior_scale = current_octave[sample[1]+1][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
    
    gradient = np.array(([posterior_scale[1,1]-previous_scale[1,1],
                          current_scale[2,1]-current_scale[0,1],
                          current_scale[1,2]-current_scale[1,0]]))/2
    
    Hessian = np.array([[posterior_scale[1,1]+previous_scale[1,1]-2*current_scale[1,1],
                        (posterior_scale[2,1]-posterior_scale[0,1]-previous_scale[2,1]+previous_scale[0,1])/4,
                        (posterior_scale[1,2]-posterior_scale[1,0]-previous_scale[1,2]+previous_scale[1,0])/4],
                       
                        [(posterior_scale[2,1]-posterior_scale[0,1]-previous_scale[2,1]+previous_scale[0,1])/4,
                        current_scale[2,1]+current_scale[0,1]-2*current_scale[1,1],
                        (current_scale[2,2]-current_scale[2,0]-current_scale[0,2]+current_scale[0,0])/4],
                        
                       [(posterior_scale[1,2]-posterior_scale[1,0]-previous_scale[1,2]+previous_scale[1,0])/4,
                        (current_scale[2,2]-current_scale[2,0]-current_scale[0,2]+current_scale[0,0])/4,
                        current_scale[1,2]+current_scale[1,0]-2*current_scale[1,1]]])
    
    invH = np.linalg.inv(Hessian)
    alpha = -np.matmul(invH, gradient[:,None])
    omega = current_scale[1,1] + np.matmul(gradient[None,:], alpha).item()/2
    return alpha, omega


# Function to interpolate keypoints

def keypoints_interpolation (dog, extrema,
                             number_scales = 3, # per octave
                             sigma_min = 0.8, # blur level in the seed image
                             delta_min = 0.5 # inter-sample distance in the seed image
                             ):
    
    candidate_keypoint = []
        
    for i in range(len(extrema)):
        print(i)
        o, s, m, n = extrema[i]
        delta0 = delta_min * 2**(o-1+1)
        repetition = 0
        coordinates = np.zeros(3)
        
        # Compute the local quadratic function
        alpha, omega = quadratic_interpolation(dog, extrema[i])
        if (np.max(np.abs(alpha)) <= 0.5):
            coordinates = np.array([delta0/delta_min * 2**((alpha[0]+s)/number_scales),
                                     delta0*(alpha[1]+m), delta0*(alpha[2]+n)])
            candidate_keypoint.append(np.append(np.concatenate((np.array([o,s,m,n]), coordinates.flatten())), omega))
        
        else:       
            while (np.max(np.abs(alpha)) > 0.5) and (repetition != 5):
            
                repetition +=1
                alpha[alpha < -0.5] = -0.5 + 10**(-15) # To correct the rounding
                alpha[alpha > -0.5] = 0.5
            
                # Compute the corresponding absolute coordinates
                coordinates = np.array([delta0/delta_min * 2**((alpha[0]+s)/number_scales),
                                        delta0*(alpha[1]+m), delta0*(alpha[2]+n)])
            
                # Update the interpolation position
                s = np.min([int(np.round(s+alpha[0]).item()), number_scales])
                m = int(np.round(m+alpha[1]).item())
                n = int(np.round(n+alpha[2]).item())
            
                # Compute the local quadratic function
                alpha, omega = quadratic_interpolation(dog, [o,s,m,n])
    
            if np.max(np.abs(alpha)) < 0.6:
                candidate_keypoint.append(np.append(np.concatenate((np.array([o,s,m,n]), coordinates.flatten())), omega))

    return candidate_keypoint


# Function to discard low contrasted candidate keypoints

def discard_low_contrasted_keypoints(candidate_keypoints, 
                                     threshold_dog = 0.015 # default value for s=3
                                     ):
    candidates = []

    for i in range(len(candidate_keypoints)):
        if np.abs(candidate_keypoints[i][-1]) >= threshold_dog:
            candidates.append(candidate_keypoints[i])
            
    return candidates


# Function to discard candidate keypoints on edges and get the SIFT keypoints

def SIFT_keypoints(dog, candidate_keypoints,
                   threshold_edge = 10):
    
    bound = (threshold_edge+1)**2/threshold_edge
    keypoints = []

    for i in range(len(candidate_keypoints)):
        
        # Compute the 2D Hessian
        sample = [int(candidate_keypoints[i][j]) for j in range(4)]
        
        current_octave = dog[sample[0]]
        previous_scale = current_octave[sample[1]-1][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
        current_scale = current_octave[sample[1]][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
        posterior_scale = current_octave[sample[1]+1][range(sample[2]-1,sample[2]+2),:][:,range(sample[3]-1,sample[3]+2)]
        
        Hessian = np.array([[posterior_scale[1,1]+previous_scale[1,1]-2*current_scale[1,1],
                            (posterior_scale[2,1]-posterior_scale[0,1]-previous_scale[2,1]+previous_scale[0,1])/4,
                            (posterior_scale[1,2]-posterior_scale[1,0]-previous_scale[1,2]+previous_scale[1,0])/4],
                           
                            [(posterior_scale[2,1]-posterior_scale[0,1]-previous_scale[2,1]+previous_scale[0,1])/4,
                            current_scale[2,1]+current_scale[0,1]-2*current_scale[1,1],
                            (current_scale[2,2]-current_scale[2,0]-current_scale[0,2]+current_scale[0,0])/4],
                            
                           [(posterior_scale[1,2]-posterior_scale[1,0]-previous_scale[1,2]+previous_scale[1,0])/4,
                            (current_scale[2,2]-current_scale[2,0]-current_scale[0,2]+current_scale[0,0])/4,
                            current_scale[1,2]+current_scale[1,0]-2*current_scale[1,1]]])
        
        # Compute the edgeness
        edgeness = np.trace(Hessian)**2 / np.linalg.det(Hessian)
        
        if edgeness < bound:
            keypoints.append(candidate_keypoints[i])
            
    return keypoints



# Function to compute the gradient at each image of the scale-space

def gradient_octaves (set_of_octaves):
    
    gradient = []
    
    for o in range(len(set_of_octaves)):
        M, N = set_of_octaves[o][0].shape
        gradient1 = []
        
        for s in range(1,(len(set_of_octaves[0])-2)): 
            partialx = np.zeros((M-2,N-2))
            partialy = np.zeros((M-2,N-2))
            for m in range(1,M-1):
                for n in range(1,N-1):
                    partialx[m-1,n-1] = (set_of_octaves[o][s][m+1][n] - set_of_octaves[o][s][m-1][n])/2
                    partialy[m-1,n-1] = (set_of_octaves[o][s][m][n+1] - set_of_octaves[o][s][m][n-1])/2
         
            gradient1.append([partialx, partialy])
            
        gradient.append(gradient1)
        
    return(gradient)



# Function to compute the keypoint reference orientation

def keypoint_orientation(keypoints, gradient,
                         lambda_ori = 1.5, # the patch is 6*lambda_ori*sigma_key wide for
                                           # a keypoint of scale sigma_key
                         number_bins = 36,
                         delta_min = 0.5,
                         threshold = 0.8):
    
    oriented_keypoints = []
    for i in range(len(keypoints)):
        
        # Check if the keypoint is distant enough from the image borders
        o_key = int(keypoints[i][0])
        delta_key = delta_min * 2**(o_key-1+1)     
        height, width = gradient[o_key][0][0].shape 
        height = height*delta_key
        width = width*delta_key
        
        s_key = int(keypoints[i][1])-1
        
        x_key = keypoints[i][5]
        y_key = keypoints[i][6]
        sigma_key = keypoints[i][4]
        
        if ((3*lambda_ori*sigma_key <= x_key <= height-3*lambda_ori*sigma_key) and
            (3*lambda_ori*sigma_key <= y_key <= width-3*lambda_ori*sigma_key)):
            
            # Initialize the orientation histogram
            hist = np.zeros(number_bins)
    
            # Accumulate samples from the normalized patch P_ori
            
            m_min = int(np.round((x_key-3*lambda_ori*sigma_key)/delta_key))
            m_max = int(np.round((x_key+3*lambda_ori*sigma_key)/delta_key))
            n_min = int(np.round((y_key-3*lambda_ori*sigma_key)/delta_key))
            n_max = int(np.round((y_key+3*lambda_ori*sigma_key)/delta_key))            
                        
            for m in range(m_min, m_max+1): 
                for n in range(n_min, n_max+1):
                    
                    # Compute the sample contribution
                    
                    diference = np.array([m*delta_key,n*delta_key])-np.array([x_key,y_key])
                    point_gradient = np.array([gradient[o_key][s_key][0][m-1][n-1],
                                               gradient[o_key][s_key][1][m-1][n-1]])
                    contribution = (np.exp(-np.linalg.norm(diference)**2/(2*(lambda_ori*sigma_key)**2))
                                    * np.linalg.norm(point_gradient))
                    
                    # Compute the arctang mod 2pi
                    arc_tan = (np.arctan2(point_gradient[0], point_gradient[1])+2*np.pi) % (2*np.pi)
                    
                    # Compute the corresponding bin index
                    bin_ori = int(np.round(number_bins/(2*np.pi) * arc_tan))

                    # Update the histogram
                    hist[bin_ori-1] = hist[bin_ori-1] + contribution
                    
            # Smooth the histogram
            
            kernel = np.ones(3) / 3
            kernel_padded = np.zeros(len(hist))
            kernel_padded[:3] = kernel
            
            # Apply six times
            repetition = 0
            
            while repetition != 6:
                hist = np.fft.ifft(np.fft.fft(hist) * np.fft.fft(kernel_padded)).real
                repetition +=1
                
            # Extract the reference orientations
            
            for k in range(1, number_bins+1):
                k_minus = (k-1) % number_bins 
                k_plus = (k+1) % number_bins 
                
                if hist[k-1] > np.max([hist[k_minus-1], hist[k_plus-1], threshold*np.max(hist)]): # -1 is for the starting in 0
                    
                    # Compute the reference orientation 
                    theta = 2*np.pi*k / number_bins
                    theta_key = theta + (np.pi/number_bins * 
                                         ((hist[k_minus-1]-hist[k_plus-1])/
                                          (hist[k_minus-1]-2*hist[k-1]+hist[k_plus-1])))
                    
                    oriented_keypoints.append(np.append(keypoints[i], theta_key))
                    
    return oriented_keypoints


# Function to compute the keypoint descriptor

def keypoint_descriptor(oriented_keypoints, gradient,
                         lambda_ori = 1.5, # the patch is 6*lambda_ori*sigma_key wide for
                                           # a keypoint of scale sigma_key
                         number_hist = 4, # the descriptor is an array of number_hist x number_hist
                                          # orientation histograms
                         number_ori = 8, # number of bins in the orientation histograms
                         lambda_descr = 6, # the Gaussian window has a standard deviation of
                                          # lamda_descr * sigma_key
                         delta_min = 0.5):
                                              
    features = []
    
    # Compute all ^x_i, ^y_j
    associated_positions = [(pos-(1+number_hist)/2) * 2*lambda_descr/number_hist for pos in range(number_hist)]
    
    # Compute ^theta_k
    hist_center = [(2*np.pi*(ori-1+1)/number_ori+2*np.pi) % (2*np.pi) for ori in range(number_ori)]
    # hist_center = [2*np.pi*(ori-1)/number_ori for ori in range(number_ori)]
    
    for i in range(len(oriented_keypoints)):
        print(i)
        
        # Check if the keypoint is distant enough from the image borders
        o_key = int(oriented_keypoints[i][0])
        delta_key = delta_min * 2**(o_key-1+1)     
        height, width = gradient[o_key][0][0].shape 
        height = height*delta_key
        width = width*delta_key
        
        s_key = int(oriented_keypoints[i][1])-1
        
        x_key = oriented_keypoints[i][5]
        y_key = oriented_keypoints[i][6]
        sigma_key = oriented_keypoints[i][4]
        theta_key = oriented_keypoints[i][8]
        
        if ((np.sqrt(2)*lambda_descr*sigma_key <= x_key <= height-np.sqrt(2)*lambda_descr*sigma_key) and
            (np.sqrt(2)*lambda_descr*sigma_key <= y_key <= width-np.sqrt(2)*lambda_descr*sigma_key)):
            
            # Initialize the array of weighted histograms
            
            histograms = [[np.zeros((number_ori)) for _ in range(number_hist)] for _ in range(number_hist)]
            
            # Accumulate samples from the normalized patch P_descr
                        
            coeff = np.sqrt(2)*lambda_descr*sigma_key
            m_min = int(np.round((x_key-coeff)/delta_key))
            m_max = int(np.round((x_key+coeff)/delta_key))
            n_min = int(np.round((y_key-coeff)/delta_key))
            n_max = int(np.round((y_key+coeff)/delta_key))            
                        
            for m in range(m_min, m_max+1): 
                for n in range(n_min, n_max+1):
                                
                    # Compute the normalized coordinates
                                
                    x_norm = (((m*delta_key-x_key)*np.cos(theta_key) +
                               (n*delta_key-y_key)*np.sin(theta_key)) / sigma_key)
                    y_norm = ((-(m*delta_key-x_key)*np.sin(theta_key) +
                              (n*delta_key-y_key)*np.cos(theta_key)) / sigma_key)
                    
                    # Verify if the sample is inside the normalized patch
                                
                    if (np.max([np.abs(x_norm), np.abs(y_norm)]) < 
                            lambda_descr*(number_hist+1)/number_hist):
                                    
                        # Compute the normalized gradient orientation
                                    
                        point_gradient = np.array([gradient[o_key][s_key][0][m-1][n-1],
                                               gradient[o_key][s_key][1][m-1][n-1]])
                        theta_norm = (np.arctan2(point_gradient[0], point_gradient[1])
                                  -theta_key+2*np.pi) % (2*np.pi)
                                
                        # Compute the total contribution of the sampla*n_e
                                    
                        diference = np.array([m*delta_key,n*delta_key])-np.array([x_key,y_key])
                        contribution = (np.exp(-np.linalg.norm(diference)**2/(2*(lambda_descr*sigma_key)**2))
                                        * np.linalg.norm(point_gradient))

                        # Update the nearest histograms and the nearest bins
                                    
                        for a in range(number_hist):
                            if np.abs(associated_positions[a]-x_norm) <= 2*lambda_descr/number_hist:
                                        
                                for b in range(number_hist):
                                    if np.abs(associated_positions[b]-y_norm) <= 2*lambda_descr/number_hist:
                                                    
                                        for k in range(number_ori):
                                            if (np.abs((hist_center[k]-theta_norm+2*np.pi) % (2*np.pi)) <
                                                2*np.pi/number_ori):
                                                            
                                                histograms[a][b][k] += ((1-number_hist/(2*lambda_descr)*
                                                                      np.abs(x_norm-associated_positions[a]))*
                                                                     (1-number_hist/(2*lambda_descr)*
                                                                      np.abs(y_norm-associated_positions[b]))*
                                                                     (1-number_ori/(2*np.pi)*
                                                                      np.abs((theta_norm-hist_center[k]+2*np.pi) % 
                                                                             (2*np.pi))) * contribution)

            # Build the feature vector from the array of weighted histograms
        
            f = np.zeros(number_hist*number_hist*number_ori)
            for a in range(number_hist): 
                for b in range(number_hist):
                    for k in range(number_ori): 
                        f[a*number_hist*number_ori+b*number_ori+k] = histograms[a][b][k]
        
        
            # Renormalize
        
            f_norm = np.linalg.norm(f)
            f_normalized = [np.min([f[l], 0.2*f_norm]) for l in range(len(f))]
        
            # Quantize to 8 bit integers
        
            f_normalized_norm = np.linalg.norm(f_normalized)
            f_integer = [np.min([np.floor(512*f_normalized[l]/f_normalized_norm), 255]) for l in range(len(f_normalized))]
        
            features.append([x_key, y_key, sigma_key, theta_key, f_integer])
    
    return features


# Function for matching points

def matching(keydes1, keydes2, 
             threshold_matching = 0.6 # relative threshold
             ):
    
    matches = []
    
    for i in range(len(keydes1)):
        
        # Find all distances to the descriptors in the second set
        distances = [np.linalg.norm(np.array(keydes1[i][-1])-np.array(keydes2[j][-1])) for j in range(len(keydes2))]
        
        # Fint the two nearest descriptors
        nearest_descriptors = np.sort(distances)[:2]
        
        # Select pair satisfying a relative threshold
        
        if nearest_descriptors[0] < threshold_matching * nearest_descriptors[1]:
            
            matches.append([keydes1[i], keydes2[np.argmin(distances)]])
    
    return matches


# Funtion to find the SIFT keypoints and descriptors of an image

def SIFT_keypoints_descriptors(image, 
                               delta = 0.5,
                               number_octaves = 4,
                               number_scales = 3, # per octave
                               sigma_min = 0.8, # blur level in the seed image
                               delta_min = 0.5, # inter-sample distance in the seed image
                               sigma_in = 0.5 # asumed blur level in the input image
                               ):
    
    # Compute the Gaussian scale-space
    set_of_octaves = digital_Gaussian_scale_space(image)

    print('LLegamos a todos los Gaussian octavas')
    
    # Compute the Difference of Gaussians
    dog = DoG(set_of_octaves)
    print('LLegamos a la diferencia de Gaussianas')
    # Find 3D discrete extrema of DoG
    extrema = extrema_DoG(dog)
    print('LLegamos a los extremos de DoG')
    # Discard low contrasted candidate keypoints
    filtered_extrema = filter_extrema_DoG(dog, extrema)
    print('LLegamos a los extremos filtrados de DoG')
    # Refine candidate keypoints location with sub-pixel precision
    interpolated_extrema = keypoints_interpolation(dog, filtered_extrema)
    
    # Filter unstable keypoints due to noise
    candidate_keypoints = discard_low_contrasted_keypoints(interpolated_extrema)
    
    # Filter unstable keypoints lying on edges
    keypoints = SIFT_keypoints(dog, candidate_keypoints)
    
    # Assign a reference orientation to each point    
    octaves_gradient = gradient_octaves(set_of_octaves)
    oriented_keypoints = keypoint_orientation(keypoints, octaves_gradient)
    print('LLegamos a los keypoints orientados')
    # Build the keypoints descriptor
    keydes = keypoint_descriptor(oriented_keypoints, octaves_gradient)
    
    return keydes


# Function to plot matching

def plot_matching(image1, image2, matching_points):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Points (x, y) in each image to connect
    points_img1 = [(np.round(matching_points[i][0][0]), np.round(matching_points[i][0][1]))
      for i in range(len(matching_points))]
    points_img2 = [(np.round(matching_points[i][1][0]), np.round(matching_points[i][1][1]))
      for i in range(len(matching_points))]


    # Show images
    ax1.imshow(image1, cmap='gray')
    ax1.axis('off')

    ax2.imshow(image2, cmap='gray')
    ax2.axis('off')

    # Plot points on each image
    x1, y1 = zip(*points_img1)
    ax1.plot(y1, x1, 'ro', markersize=8)

    x2, y2 = zip(*points_img2)
    ax2.plot(y2, x2, 'ro', markersize=8)

    # Connect corresponding points between the two images
    for (x1_p, y1_p), (x2_p, y2_p) in zip(points_img1, points_img2):
        con = patches.ConnectionPatch(
            xyA=(y2_p, x2_p), coordsA=ax2.transData,
            xyB=(y1_p, x1_p), coordsB=ax1.transData,
            arrowstyle='-', color='blue', linewidth=2)
        fig.add_artist(con)

    plt.tight_layout()
    plt.show()

def plot_SAR_matching(image1, image2, matching_points, maximum=0.25):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Points (x, y) in each image to connect
    points_img1 = [(np.round(matching_points[i][0][0]), np.round(matching_points[i][0][1]))
      for i in range(len(matching_points))]
    points_img2 = [(np.round(matching_points[i][1][0]), np.round(matching_points[i][1][1]))
      for i in range(len(matching_points))]


    # Show images
    ax1.imshow(image1, cmap='gray', vmin=0, vmax=maximum)
    ax1.axis('off')

    ax2.imshow(image2, cmap='gray', vmin=0, vmax=maximum)
    ax2.axis('off')

    # Plot points on each image
    x1, y1 = zip(*points_img1)
    ax1.plot(y1, x1, 'ro', markersize=8)

    x2, y2 = zip(*points_img2)
    ax2.plot(y2, x2, 'ro', markersize=8)

    # Connect corresponding points between the two images
    for (x1_p, y1_p), (x2_p, y2_p) in zip(points_img1, points_img2):
        con = patches.ConnectionPatch(
            xyA=(y2_p, x2_p), coordsA=ax2.transData,
            xyB=(y1_p, x1_p), coordsB=ax1.transData,
            arrowstyle='-', color='blue', linewidth=1)
        fig.add_artist(con)

    plt.tight_layout()
    plt.show()

def sift_algorithm(image1, image2, umbral_matching=0.8):
    print('Iniciando SIFT...')
    # Obtener keypoints y descriptores SIFT para ambas imágenes
    keydes1 = SIFT_keypoints_descriptors(image1[:,:,1])
    keydes2 = SIFT_keypoints_descriptors(image2[:,:,1])

    print('Keypoints y descriptores obtenidos.')
    print(f'Número de keypoints en la imagen 1: {len(keydes1)}')
    print(f'Número de keypoints en la imagen 2: {len(keydes2)}')

    # Emparejar descriptores entre las dos imágenes
    matching_points = matching(keydes1, keydes2, threshold_matching=umbral_matching)

    print(f'Número de puntos coincidentes: {len(matching_points)}')

    return matching_points


def sift_opencv(imagen1, imagen2, umbral_matching=0.8):
    # Convertir a escala de grises

    # Crear el objeto SIFT
    sift = cv2.SIFT_create()

    # Detectar keypoints y descriptores
    kp1, des1 = sift.detectAndCompute(imagen1, None)
    kp2, des2 = sift.detectAndCompute(imagen2, None)

    # Emparejar descriptores con FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Selección de buenos matches
    good = []
    for m, n in matches:
        if m.distance < umbral_matching * n.distance:
            good.append(m)

    # Dibujar los matches
    resultado = cv2.drawMatches(imagen1, kp1, imagen2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return resultado

def imagen_keypoints(image, points, color='red', markersize=8):

    fig, ax = plt.subplots(figsize=(image.shape[1]/100, image.shape[0]/100), dpi=100)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    x, y = zip(*points)
    ax.plot(y, x, 'o', color=color, markersize=markersize)
    fig.tight_layout(pad=0)

    # Convertir el canvas de Matplotlib a imagen NumPy
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_array