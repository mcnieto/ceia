import numpy as np
import cv2 as cv


if __name__ == "__main__":

    # Se cargan desde archivo clasificadores pre-entrenados
    face_cascade  = cv.CascadeClassifier('c:\Documentos\AI\computer_vision_1\TPs\TP6\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
    eye_cascade   = cv.CascadeClassifier('c:\Documentos\AI\computer_vision_1\TPs\TP6\opencv\data\haarcascades\haarcascade_eye.xml')
    smile_cascade = cv.CascadeClassifier('c:\Documentos\AI\computer_vision_1\TPs\TP6\opencv\data\haarcascades\haarcascade_smile.xml')
    
    cap = cv.VideoCapture(0)
    cv.namedWindow("frame")
    key = 0
    
    # Grabación de video
    # Obtiene la resolución por defecto del frame. La misma depende del systema.
    # Luego, se convierte la convierte de float a integer. 
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    out = cv.VideoWriter('c:\Documentos\AI\computer_vision_1\TPs\TP6\outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    
    while(1):
        ret, frame = cap.read()
        
        if ret == True:
            
            key = cv.waitKey(1) & 0XFF
            
            if key == 27:
                break
            else:
                # Conversión a escala de grises y ecualizacion
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                gray_frame = cv.equalizeHist(gray_frame)
                
                #Llamada al clasificador de Haar (AdaBoost)
                faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)
                # Detección de rostro
                for (x,y,w,h) in faces:
                    # Recuadrado con rectángulo amarillo
                    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                    # Definición de las ROIs en la imagen gris y color
                    roi_gray = gray_frame[y:y+h, x:x+w] 
                    roi_color = frame[y:y+h, x:x+w] 
                    
                    # Detección de ojos sobre el rostro hallado
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.8, 2)
                    # Recuadrado con circulos azules
                    rad = 30
                    for (ex,ey,ew,eh) in eyes:
                        cv.circle(roi_color,(ex + rad, ey + rad),rad,(255,0,0),2)
                        
                    # Detección de sonrisas sobre el rostro hallado
                    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 10)  
                    # Recuadrado con rectángulo rojo
                    for (sx,sy,sw,sh) in smiles:
                        cv.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
                        
                cv.imshow('frame',frame)         
                out.write(frame)
    
        else:
            break
    
    cv.destroyAllWindows()
    cap.release()
    out.release()
