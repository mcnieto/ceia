import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def img_gradient(img_gray):
    # Suavizado Gaussiano
    blur = cv.GaussianBlur(img_gray,(5,5),0)
    
    # Gradientes
    # Se aplica Sobel en eje x en 'float32' y luego se lo convierte a de nuevo a 8-bit para evitar overflow
    sobelx_64 = cv.Sobel(blur,cv.CV_32F,1,0,ksize=3)
    absx_64 = np.absolute(sobelx_64)
    sobelx_8u1 = absx_64/absx_64.max()*255
    sobelx_8u = np.uint8(sobelx_8u1)
    
    # Se aplica Sobel en eje y en 'float32' y luego se lo convierte a de nuevo a 8-bit para evitar overflow
    sobely_64 = cv.Sobel(blur,cv.CV_32F,0,1,ksize=3)
    absy_64 = np.absolute(sobely_64)
    sobely_8u1 = absy_64/absy_64.max()*255
    sobely_8u = np.uint8(sobely_8u1)

    # Cálculo de la magnitud de los gradientes y escalado a uint8
    mag = np.hypot(sobelx_8u, sobely_8u)
    mag = mag/mag.max()*255
    mag = np.uint8(mag)

    # Cálculo de la dirección de los gradientes y pasaje a grados
    theta = np.arctan2(sobely_64, sobelx_64)
    angle = np.rad2deg(theta)
    return mag, angle


def gradient_angle_detector(img, central_angle = 30, rage_angle = 60, gradient_mag_threeshold = 100, color = (255,0,0)):
    img_gray = cv.cvtColor(img_BGR,cv.COLOR_BGR2GRAY)
    mag, angle = img_gradient(img_gray)
    init_angle = central_angle - rage_angle/2
    end_angle = central_angle + rage_angle/2
    
    angle_mask = (angle >= init_angle) & (angle <= end_angle) 
    mag_mask = mag >= gradient_mag_threeshold
    img[angle_mask & mag_mask] = color
    return img

#%%
img_BGR = cv.imread('c:\Documentos\AI\computer_vision_1\TPs\TP2\metalgrid.jpg',cv.IMREAD_COLOR)
img_RGB = cv.cvtColor(img_BGR,cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img_BGR,cv.COLOR_BGR2GRAY)

cv.imshow('imagen en escala de grises',img_gray)
cv.imwrite('c:\Documentos\AI\computer_vision_1\TPs\TP3\img_gray.jpg', img_gray) 
cv.waitKey(0)
cv.destroyAllWindows()

#%%
mag, angle = img_gradient(img_gray)

fig, ax = plt.subplots(1, 1, figsize=(20,8)) 
im0 = ax.imshow(mag,aspect='auto', cmap='gray', vmin=0, vmax=255.0)
ax.set_title("Gradiente de la imagen [Magnitud]")
ax.axis('off')
plt.colorbar(im0)
plt.show()
fig.savefig('img_gradients_mag.jpg') 

fig, ax = plt.subplots(1, 1, figsize=(20,8)) 
im1 = ax.imshow(angle,aspect='auto', cmap='Spectral', vmin=-180.0, vmax=180.0)
ax.set_title("Gradiente de la imagen [Angulo]")
ax.axis('off')
plt.colorbar(im1)
plt.show()
fig.savefig('img_gradients_angle.jpg') 

#%%
initial_angle = -180+1*45
angle_span=90.0
mag_threeshold = 100.0
color=(255,0,0)
alpha=1.0      
end_angle = initial_angle + angle_span

angle_mask = (angle >= initial_angle) & (angle <= end_angle) 
mag_mask = mag >= mag_threeshold
img_colored = img.copy()
img_colored[angle_mask&mag_mask] = color
img_colored[~(angle_mask&mag_mask),1] = img[~(angle_mask&mag_mask),1]*alpha
img_colored[~(angle_mask&mag_mask),2] = img[~(angle_mask&mag_mask),2]*alpha
    
plt.figure(figsize=(16,12))
plt.imshow(img_colored,aspect='auto', interpolation='nearest')
plt.title(f"Ángulos de {initial_angle} a {end_angle} c/ umbral de magnitud {mag_threeshold}")
plt.axis('off')
plt.show()

#%%
img = img_RGB.copy()
central_angle = 30
ragen_angle = 30
mag_threeshold = 100.0
color=(255,0,0)
init_angle = central_angle - ragen_angle/2
end_angle = central_angle + ragen_angle/2

angle_mask = (angle >= init_angle) & (angle <= end_angle) 
mag_mask = mag >= mag_threeshold
img[angle_mask & mag_mask] = color
return img
 
plt.figure(figsize=(16,12))
plt.imshow(img,aspect='auto', interpolation='nearest')
plt.title(f"Ángulos de {initial_angle} a {end_angle} c/ umbral de magnitud {mag_threeshold}")
plt.axis('off')
plt.show()

#%%
img_grandient_angle_det = gradient_angle_detector(img_RGB, central_angle = 30, rage_angle = 60, gradient_mag_threeshold = 100, color = (0,255,0))

colors = np.zeros((6,3,1), dtype = uint8)
colors[0,:,0] = [255,0,0]
colors[1,:,0] = [255,127,0]
colors[2,:,0] = [255,255,0]
colors[3,:,0] = [0,255,0]
colors[4,:,0] = [0,0,255]
colors[5,:,0] = [75,0,130]

central_angle_array = np.array[30, 90, 150, -30, -90, -150]

img_grandient_angle_det = img_RGB.copy()
for i in 6:
    img_grandient_angle_det = gradient_angle_detector(img_grandient_angle_det, 
                                                        central_angle = central_angle_array[i], 
                                                        rage_angle = 60, 
                                                        gradient_mag_threeshold = 100, 
                                                        color = colors[i,:,0])

cv.imshow('img',img_grandient_angle_det)
cv.waitKey(0)
cv.destroyAllWindows()

plt.figure(figsize=(16,12))
plt.imshow(img_grandient_angle_det,aspect='auto', interpolation='nearest')
plt.show()
