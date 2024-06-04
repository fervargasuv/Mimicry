import cv2
import mediapipe as mp
import csv
import os

def generar_landmarks(video_path, output_folder):
    # Cargar el video
    cap = cv2.VideoCapture(video_path)
    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return

    # Inicializar MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Leer el video frame por frame
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir el frame a RGB (MediaPipe trabaja con imágenes en RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame con MediaPipe FaceMesh
        results = mp_face_mesh.process(rgb_frame)

        # Inicializar una lista de landmarks con ceros
        landmarks_px = [(0, 0)] * 468  # MediaPipe FaceMesh tiene 468 landmarks por defecto

        # Obtener los landmarks faciales si están disponibles
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convertir los landmarks de MediaPipe a coordenadas en píxeles del frame
                landmarks_px = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

        # Guardar las coordenadas de los landmarks en un archivo CSV por frame
        output_file = f'{output_folder}/{frame_count}.csv'
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Landmark', 'X', 'Y'])
            for idx, landmark in enumerate(landmarks_px):
                csv_writer.writerow([f'{idx}', landmark[0], landmark[1]])

        # Incrementar el contador de frames
        frame_count += 1

    # Liberar el objeto VideoCapture
    cap.release()

def procesar_videos_en_carpeta(root_folder):
    # Recorrer todas las carpetas dentro de la carpeta raíz
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        # Verificar si es una carpeta
        if os.path.isdir(folder_path):
            # Buscar el archivo de video dentro de la carpeta
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.avi'):
                    video_path = os.path.join(folder_path, file_name)
                    output_folder = folder_path
                    print(f'Procesando video: {video_path}')
                    generar_landmarks(video_path, output_folder)
                    print(f'Procesamiento de {video_path} completado.')

# Ruta a la carpeta raíz que contiene todas las carpetas de videos
root_folder = 'SEWA DB v0.2'

procesar_videos_en_carpeta(root_folder)
