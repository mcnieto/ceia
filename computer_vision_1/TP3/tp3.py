import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def distance_meter (cicles_array):
    left_eyes = np.zeros(np.uint8(cicles_array.shape[1]/2), dtype = int)
    right_eyes = np.zeros(np.uint8(cicles_array.shape[1]/2), dtype = int)
    # Ordena los círculos de acuerdo a su posición en el eje y de la imagen (columnas de la matriz)
    order_cicles = np.argsort(cicles_array[0,:,1])
    # Asocia en pares dos círculos ubicados en aproximadamente la misma posición en el eje y de la imagen
    # cada par se compone por ojo izq y ojo derecho
    odd = np.arange(0,cicles_array.shape[1],2)
    even = odd + 1
    left_eyes = cicles_array[0,order_cicles[odd],0]
    right_eyes = cicles_array[0,order_cicles[even],0]
    # Resta las posición x en la imagen correspondiente a cada par iqz-der
    distance =np.absolute(np.int16(right_eyes-left_eyes))
    return distance

# Carga de imagen
img_BGR = cv.imread('c:\Documentos\AI\computer_vision_1\TPs\TP3\eyes.jpg')
img_RGB = cv.cvtColor(img_BGR,cv.COLOR_BGR2RGB)
plt.imshow(img_RGB)
plt.show()

img_gray = cv.cvtColor(img_BGR,cv.COLOR_BGR2GRAY)

# Suavizado de la imagen con un filtro de mediana
# img_gray = cv.GaussianBlur(eye_gray,ksize=(3,3),sigmaX=1)
img_gray = cv.medianBlur(img_gray,5)

cv.imshow('imagen en escala de grises suavizada',img_gray)
cv.imwrite('c:\Documentos\AI\computer_vision_1\TPs\TP3\filtered_gray_eyes.jpg', img_gray) 
cv.waitKey(0)
cv.destroyAllWindows()

#%%
img_gray_iris = img_gray.copy()
iris_circles = cv.HoughCircles(img_gray_iris,       # imagen en grises
                                cv.HOUGH_GRADIENT,  # método
                                1,                  # flag de resolución del acumulador
                                90,                # dist mín entre centros de círculos
                                param1=200,         # umbral alto para Cany
                                param2=15,          # umbral del acumulador
                                minRadius=28,       # radio_min
                                maxRadius=31)       # radio_max
iris_circles = np.uint16(np.around(iris_circles))
for i in iris_circles[0,:]:
    # draw the outer circle
    cv.circle(img_gray_iris,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(img_gray_iris,(i[0],i[1]),2,(255,255,255),3)
    
cv.imshow('circulos detectados en los iris de los ojos',img_gray_iris)
cv.imwrite('c:\Documentos\AI\computer_vision_1\TPs\TP3\eyes_gray_iris_cicles.jpg', img_gray_iris) 
cv.waitKey(0)
cv.destroyAllWindows()

distance = distance_meter(iris_circles)
print('\nLas distancias medidas en pixeles entre los iris de cada par de ojos son las siguientes:\n')
for i in range(distance.shape[0]):
    print('Par nro. {} : {}'.format(i+1, distance[i]))

#%%
img_gray_pupils = img_gray.copy()
pupils_circles = cv.HoughCircles(img_gray_pupils,   # imagen en grises
                                cv.HOUGH_GRADIENT,  # método
                                1,                  # flag de resolución del acumulador
                                90,                 # dist mín entre centros de círculos
                                param1=160,         # umbral alto para Cany
                                param2=15,          # umbral del acumulador
                                minRadius=10,       # radio_min
                                maxRadius=15)       # radio_max
                                
pupils_circles = np.uint16(np.around(pupils_circles))
for i in pupils_circles[0,:]:
    # draw the outer circle
    cv.circle(img_gray_pupils,(i[0],i[1]),i[2],(255,255,0),2)
    # draw the center of the circle
    cv.circle(img_gray_pupils,(i[0],i[1]),2,(255,255,0),3)
cv.imshow('círculos detectados en los iris de los ojos',img_gray_pupils)
cv.imwrite('c:\Documentos\AI\computer_vision_1\TPs\TP3\eyes_gray_pupils_cicles.jpg', img_gray_pupils) 
cv.waitKey(0)
cv.destroyAllWindows()

distance = distance_meter(pupils_circles)
print('\nLas distancias medidas en pixeles entre las pulilas de cada par de ojos son las siguientes:\n')
for i in range(distance.shape[0]):
    print('Par nro. {} : {}'.format(i+1, distance[i]))