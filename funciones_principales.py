import numpy as np
import cv2


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


def aplicar_cruces(imagen,grises=True):
    if grises is True:
        H,W = imagen.shape[:2]
        imagen_cruces = np.zeros_like(imagen)
        for i in range(H - 1):
            for j in range(W - 1):
                if cambio_signo(imagen[i, j],imagen[i, j + 1]) or cambio_signo(imagen[i, j],imagen[i+1, j]):
                  imagen_cruces[i, j] = 255
                else:
                    imagen_cruces[i, j] = 0
    else:
        H,W, C = imagen.shape
        imagen_cruces = np.zeros_like(imagen)
        for i in range(H - 1):
            for j in range(W - 1):
                for c in range(C):
                    if cambio_signo(imagen[i, j, c],imagen[i, j + 1, c]) or cambio_signo(imagen[i, j, c],imagen[i+1, j, c]):
                        imagen_cruces[i, j, c] = 255
                    else:
                        imagen_cruces[i, j, c] = 0
    imagen_cruces = imagen_cruces.astype(np.uint8)
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

def Leclerc(valor,sigma_sensibilidad):
    division = -(valor**2) / (sigma_sensibilidad**2)
    resultado = np.exp(division)

    return resultado

def Lorentz(valor,sigma_sensibilidad):
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

    



