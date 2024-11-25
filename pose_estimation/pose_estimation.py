import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Linea 4: Esta línea accede al módulo drawing_utils de MediaPipe, 
#          que proporciona funciones para dibujar anotaciones en las imágenes, 
#          como los puntos clave y las conexiones entre ellos para representar 
#          la pose detectada. Asigna este módulo a la variable mp_drawing para 
#          facilitar su uso posterior.

# Linea 5: Esta línea accede al módulo pose de MediaPipe, que contiene las funciones 
#          y modelos necesarios para la detección de poses humanas. Asigna este módulo a la 
#          variable mp_pose, lo que permite crear y utilizar un modelo de detección de pose 
#          en el script.


def calcute_angle(a, b, c):
    """
    Calcula el angulo que existe entre tres puntos.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[-1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle
    
def calcular_angulo_3d(a, b, c):
    """Calcula el ángulo entre tres puntos 3D (en grados), donde `b` es el vértice."""
    # Crear vectores AB y BC
    ab = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    ab_u = vector_unitario(ab)
    bc_u = vector_unitario(bc)

    # Calcular el producto punto y el ángulo
    angulo_radianes = np.arccos(np.clip(np.dot(ab_u, bc_u), -1.0, 1.0))
    angulo_grados = np.degrees(angulo_radianes)
    return angulo_grados

def vector_unitario(vector):
    """Devuelve el vector unitario de un vector."""
    return vector / np.linalg.norm(vector)

# Deteccion de video
cap = cv2.VideoCapture(0)

# Global vars
counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Reordenar las matrices de colores a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Detección
        results = pose.process(image)
        
        # Reordenar las matrices de colores a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extraer puntos del cuerpo
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Extraer valores en z de puntos 11 y 12
            point_eleven = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
            #point_twelve = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
            print("Hombro izquierdo: {}".format(point_eleven))


            #print("Hombro derecho: {}".format(point_twelve))
            
            # Extraer puntos: 11, 13, 15
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Extraer puntos: 23, 11, 13
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # Calcular angulo
            angle_one = calcute_angle(shoulder, elbow, wrist)
            angle_two = calcute_angle(ankle, shoulder, elbow)
            
            # Visualizar
            cv2.putText(
                image, 
                str(angle_one),
                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0), 2, cv2.LINE_AA
            )
            
            cv2.putText(
                image, 
                str(angle_two),
                tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0), 2, cv2.LINE_AA
            )
            
            # Curl counter logic
            if angle_one > 160:
                stage = "down"
            if angle_one < 30 and stage == 'down':
                stage = 'up'
                counter += 1
                print(counter)
            
            #print(landmarks)
            
        except:
            pass
        
        # Renderizar detección
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=3)
        )
        
        cv2.imshow('Mediapipe Feed', image),
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    print(len(landmarks))
    
cap.release()
cv2.destroyAllWindows()