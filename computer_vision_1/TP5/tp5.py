#----------------------------------------------------------------------------------------
### Descripción de la aplicación:
# Esta aplicación implementa el algoritmo de seguimiento de objetos CamShift, a partir de los
# frames capturas por la webcam. Permite al usuario definir la región de interés en
# el video (ROI), y cambiarla en forma dinámica cuando se desee. Para ello admite un modo de 
# operación llamado "InputMode".
# La aplicación también registra en un video todos frames capturados durante la prueba.



# ### Instrucciones de uso:
# Al ejecutarse la aplicación se mostrará en pantalla el video capturado por la webcam. 
# Para entrar "InputMode" para definir el ROI, se debe realizar los siguientes pasos:
# - Dar click derecho en cualquier parte de la imagen de video.
# - Presionar la tecla 'i'. Al hacer esto, el frame de video se detendrá, manteniendo en 
# pantalla la ultima imagen capturada por la webcam.
# - Seleccionar sobre la imagen 4 puntos que un rectangulo sobre el objeto que se desea seguir.
# Se observará q al hacer cada click sobre la imagen se dibujan pequeños círculos verdes que
# representan los vértices del rectángulo que define ROI.
# - Una vez seleccionado los 4 puntos anteriores, presionar cualquier tecla para salir de 
# "InputMode"

# A continuación se verá sobre la imagen de video un rectángulo, cuyo tamaño variará 
# recuadrando siempre al objeto de interés.
# - Si se desea redefinir la ROI para seguir un objeto diferente, se debe entrar al modo 
# "InputMode", presionando la tecla 'i', siguiendo nuevamente los pasos anteriores.
# - Si se desea finalirzar la ejecución, presionar la tecla "Esc", para salir.

# Al finalizar la ejecución quedará grabado en video todos los frames capturos durante la 
# prueba. El mismo tendrá formato MJPG y se guardará bajo el nombre "video_tp5.avi".
#----------------------------------------------------------------------------------------

import numpy as np
import cv2 as cv

# Funcion de configuracion de ROI
def selectROI(event, x, y, flags, param):
    global frame, roiPts, inputMode

    if inputMode and event == cv.EVENT_LBUTTONDOWN and len(roiPts)<4:
        roiPts.append((x,y))
        cv.circle(frame, (x,y), 4,(0,255,0),2)
        cv.imshow("frame",frame)


# Funcion principal
if __name__ == "__main__":

    # Inicializacion de variables
    global frame, roiPts, inputMode, video
    
    # frame a procesar
    # lista de puntos que definen la region de interes en el video
    frame = None        # Current frame of the video being processed
    roiPts = []         # List of points corresponding to the Region of Interest in video
    inputMode = False   # Indicate whether or not we're selecting the object we want to track
    
    cap = cv.VideoCapture(0)
    
    cv.namedWindow("frame")
    cv.setMouseCallback("frame", selectROI)
    
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,1)
    track_window = []
    key = 0
    
    # Grabación de video
    # Obtiene la resolución por defecto del frame. La misma depende del systema.
    # Luego, se convierte la convierte de float a integer. 
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    out = cv.VideoWriter('c:\Documentos\AI\computer_vision_1\TPs\TP5\outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    
    while(1):
        ret, frame = cap.read()
        
        if ret == True:
            
            key = cv.waitKey(1) & 0XFF
    
            if key == ord("i"):
                inputMode = True
                roiPts = []
    
                while len(roiPts) <4 :
                    cv.imshow("frame",frame)
                    cv.waitKey(0)
                
                inputMode = False
                
                roiPts = np.array(roiPts)
                s = roiPts.sum(axis=1)
                tl = roiPts[np.argmin(s)]
                br = roiPts[np.argmax(s)]
                track_window = (tl[0],tl[1],br[0],br[1])
                roi = frame[tl[1]:br[1],tl[0]:br[0]]
                hsv_roi = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
                mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                roi_hist = cv.calcHist([hsv_roi],[0],mask, [180],[0,180])
                cv.normalize(roi_hist,roi_hist, 0,255,cv.NORM_MINMAX)
            
            if len(track_window) == 4:
                hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
                ret, track_window = cv.CamShift(dst, track_window, term_crit)
                x,y,w,h = track_window
                cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                
            if key == 27:
                break
            
            cv.imshow("frame",frame)
            out.write(frame)
            
        else:
            break
    
    cv.destroyAllWindows()
    cap.release()
    out.release()

#%%
if __name__ == "__main__":
    main()
