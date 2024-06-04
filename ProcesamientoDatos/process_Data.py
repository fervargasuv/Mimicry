import csv
import numpy as np
import pandas as pd
import os
import arff
from collections import defaultdict
import random
from sklearn.preprocessing import MinMaxScaler,StandardScaler
class LandmarksReader:
    def __init__(self):
        pass

    def read_landmarks_frames(self, participante, frame_inicio, frame_fin):
        lista_frames = []  # Lista para almacenar todos los frames de un participante
        lista_yaw = []  # Lista para almacenar el yaw de cada frame
        lista_pitch = []  # Lista para almacenar el pitch de cada frame
        lista_roll = []  # Lista para almacenar el roll de cada frame
        participante = participante.strip('"')
        ruta_carpeta = os.path.join('C:/Users/nekos/OneDrive/Documentos/Tesis/Codigo tesis/Full_Videos_Frames', participante)
        archivos = os.listdir(ruta_carpeta)
        
        for archivo in archivos:
            if 'Landmarks' in archivo:
                ruta_subcarpeta = os.path.join(ruta_carpeta, archivo)
                break
        
        subcarpeta = os.listdir(ruta_subcarpeta)

        for archivo in subcarpeta:
            if archivo.endswith('.txt'):
                frame_actual = int(archivo.split('.')[0])
                if frame_inicio <= frame_actual <= frame_fin:
                    with open(os.path.join(ruta_subcarpeta, archivo), 'r') as archivotxt:
                        lineas = archivotxt.readlines()
                        # Obtener yaw, pitch y roll de la primera línea del archivo
                        yaw, pitch, roll = map(float, lineas[0].strip().split())
                        lista_yaw.append(yaw)
                        lista_pitch.append(pitch)
                        lista_roll.append(roll)
                        coordenadas = lineas[2].strip().split()
                        datos_frame = []  # Lista para almacenar las coordenadas de un frame
                        for i in range(0, len(coordenadas), 2):
                            x = float(coordenadas[i])
                            y = float(coordenadas[i+1])
                            datos_frame.append((x, y))  # Almacena las coordenadas como tuplas (x, y)
                        lista_frames.append(datos_frame)  # Agrega las coordenadas del frame a la lista
        return lista_frames, lista_yaw, lista_pitch, lista_roll
    

    def extract_mimetism_cases(self):
        casos_mimetismo = []  # Lista para almacenar los casos de mimetismo

        # Obtener la lista de archivos CSV con información de mimetismo
        ruta_mimetismo = 'C:/Users/nekos/OneDrive/Documentos/Tesis/Codigo tesis/Mimicry_Catalogue'
        archivos_csv = os.listdir(ruta_mimetismo)

        for archivo_csv in archivos_csv:
            if archivo_csv.endswith('.csv') and 'EoM_C' in archivo_csv:
                ruta_archivo = os.path.join(ruta_mimetismo, archivo_csv)
                df = pd.read_csv(ruta_archivo)  # Leer el archivo CSV
                # Filtrar los casos por longitud de frames menor o igual a 300
                df_filtrado = df[df['last_frame'] - df['first_frame'] + 1 <= 300]
                # Agregar los datos relevantes a la lista de casos de mimetismo
                for caso in df_filtrado[['subject1_source', 'subject2_source', 'first_frame', 'last_frame']].values.tolist():
                    inicio = caso[2]
                    fin = caso[3]
                    longitud_actual = fin - inicio + 1
                    
                    # Si la longitud es menor a 200, ajustar el final para alcanzar 200 frames
                    if longitud_actual < 200:
                        frames_faltantes = 200 - longitud_actual
                        caso[3] += frames_faltantes
                    # Si la longitud está entre 200 y 300, reducir al intervalo de 200 frames
                    elif longitud_actual > 200 and longitud_actual <= 300:
                        caso[3] = inicio + 199  # Tomar los primeros 200 frames
                    casos_mimetismo.append(caso)

        return casos_mimetismo





    def contar_frames_carpeta(self,participante):
        # Obtener la lista de archivos en la carpeta
        participante = participante.strip('"')
        ruta_carpeta = os.path.join('C:/Users/nekos/OneDrive/Documentos/Tesis/Codigo tesis/Full_Videos_Frames', participante)
        archivos = os.listdir(ruta_carpeta)
        for archivo in archivos:
            if 'Landmarks' in archivo:
                ruta_subcarpeta = os.path.join(ruta_carpeta, archivo)
                break
        subcarpeta = os.listdir(ruta_subcarpeta)
        # Filtrar solo los archivos de texto (con extensión .txt)
        archivos_txt = [archivo for archivo in subcarpeta if archivo.endswith('.txt')]
        # Contar la cantidad de archivos de texto
        total_frames = len(archivos_txt)
        return total_frames
    
    
    def generar_casos_no_mimetismo(self,longitud_mimetismo, total_frames, inicio_mimetismo, fin_mimetismo):
        # Definir límites para la generación aleatoria
        limite_inferior = 100
        limite_superior = total_frames - longitud_mimetismo

        # Generar un rango aleatorio que no se superponga con el caso de mimetismo
        inicio_no_mimetismo = random.randint(limite_inferior, limite_superior)
        fin_no_mimetismo = inicio_no_mimetismo + longitud_mimetismo

        # Verificar si hay solapamiento y ajustar los límites si es necesario
        while (inicio_no_mimetismo <= fin_mimetismo and fin_no_mimetismo >= inicio_mimetismo):
            inicio_no_mimetismo = random.randint(limite_inferior, limite_superior)
            fin_no_mimetismo = inicio_no_mimetismo + longitud_mimetismo
        return inicio_no_mimetismo, fin_no_mimetismo
    
    def guardar_casos_mimetismo_en_csv(self,casos_mimetismo, ruta_salida):
        # Encabezados para el archivo CSV
        encabezados = ['Subject1_Source', 'Subject2_Source', 'First_Frame', 'Last_Frame']

        # Escribir la lista de casos en el archivo CSV
        with open(ruta_salida, mode='w', newline='', encoding='utf-8') as archivo_csv:
            writer = csv.writer(archivo_csv)
            writer.writerow(encabezados)  # Escribir los encabezados
            writer.writerows(casos_mimetismo)  # Escribir los casos de mimetismo

        print(f"Se guardaron los casos de mimetismo en el archivo CSV: {ruta_salida}")
    
    def formar_matriz_frames_landmarks(self,person1, person2):
        num_frames = len(person1)
        num_landmarks = len(person1[0])  # Suponiendo que todos los frames tienen la misma cantidad de landmarks
        matriz = np.zeros((num_frames, 2, num_landmarks, 2))  # Matriz inicializada con ceros
        
        for i in range(num_frames):
            for j in range(num_landmarks):
                matriz[i, 0, j] = person1[i][j]  # Puntos xy de la persona 1 en el frame i y landmark j
                matriz[i, 1, j] = person2[i][j]  # Puntos xy de la persona 2 en el frame i y landmark j

        return matriz

    def min_max_scale_coordinates(self,data):
        scaled_data = []
        
        for sublist in data:
            # Separar las coordenadas x e y en listas separadas
            x_coords = [coord[0] for coord in sublist]
            y_coords = [coord[1] for coord in sublist]

            # Crear un MinMaxScaler para las coordenadas x
            scaler_x = MinMaxScaler()
            scaler_x.fit(np.array(x_coords).reshape(-1, 1))  # Ajustar el scaler a las coordenadas x
            x_scaled = scaler_x.transform(np.array(x_coords).reshape(-1, 1))  # Escalar las coordenadas x

            # Crear un MinMaxScaler para las coordenadas y
            scaler_y = MinMaxScaler()
            scaler_y.fit(np.array(y_coords).reshape(-1, 1))  # Ajustar el scaler a las coordenadas y
            y_scaled = scaler_y.transform(np.array(y_coords).reshape(-1, 1))  # Escalar las coordenadas y

            # Combinar las coordenadas x e y escaladas en pares nuevamente
            scaled_sublist = [(x_scaled[i][0], y_scaled[i][0]) for i in range(len(sublist))]
            scaled_data.append(scaled_sublist)

        return scaled_data


reader = LandmarksReader()
casos_mimetismo = reader.extract_mimetism_cases()
# Suponiendo que 'casos_mimetismo' es la lista que quieres guardar y 'ruta_salida' es la ruta del archivo CSV de salida
ruta_salida_csv = 'casos_mimetismo.csv'
reader.guardar_casos_mimetismo_en_csv(casos_mimetismo, ruta_salida_csv)







matrices_mimetismo=[]
matrices_no_mimetismo=[]

scaler=StandardScaler()
# Recorrer los casos de mimetismo en casos_mimetismo
for caso in casos_mimetismo:
    persona1, persona2, inicio, fin = caso

    # Obtener los landmarks, yaw, pitch y roll de la persona 1
    landmarks_p1, yaw_p1, pitch_p1, roll_p1 = reader.read_landmarks_frames(persona1, inicio, fin)

    landmarks_p1=reader.min_max_scale_coordinates(landmarks_p1)
    yaw_p1=scaler.fit_transform(np.array(yaw_p1).reshape(-1, 1))
    pitch_p1=scaler.fit_transform(np.array(pitch_p1).reshape(-1, 1))
    roll_p1=scaler.fit_transform(np.array(roll_p1).reshape(-1, 1))
    
    # Obtener los landmarks, yaw, pitch y roll de la persona 2
    landmarks_p2, yaw_p2, pitch_p2, roll_p2 = reader.read_landmarks_frames(persona2, inicio, fin)

    landmarks_p2=reader.min_max_scale_coordinates(landmarks_p2)
    yaw_p2=scaler.fit_transform(np.array(yaw_p2).reshape(-1, 1))
    pitch_p2=scaler.fit_transform(np.array(pitch_p2).reshape(-1, 1))
    roll_p2=scaler.fit_transform(np.array(roll_p2).reshape(-1, 1))
    

    total_frames = reader.contar_frames_carpeta(persona1)
    inicio_nm, fin_nm = reader.generar_casos_no_mimetismo(200, total_frames, inicio, fin)

    landmarks_p1_nm, yaw_p1_nm, pitch_p1_nm, roll_p1_nm = reader.read_landmarks_frames(persona1, inicio_nm, fin_nm)

    landmarks_p1_nm=reader.min_max_scale_coordinates(landmarks_p1_nm)
    yaw_p1_nm=scaler.fit_transform(np.array(yaw_p1_nm).reshape(-1, 1))
    pitch_p1_nm=scaler.fit_transform(np.array(pitch_p1_nm).reshape(-1, 1))
    roll_p1_nm=scaler.fit_transform(np.array(roll_p1_nm).reshape(-1, 1))

    landmarks_p2_nm, yaw_p2_nm, pitch_p2_nm, roll_p2_nm = reader.read_landmarks_frames(persona2, inicio_nm, fin_nm)
    
    landmarks_p2_nm=reader.min_max_scale_coordinates(landmarks_p2_nm)
    yaw_p2_nm=scaler.fit_transform(np.array(yaw_p2_nm).reshape(-1, 1))
    pitch_p2_nm=scaler.fit_transform(np.array(pitch_p2_nm).reshape(-1, 1))
    roll_p2_nm=scaler.fit_transform(np.array(roll_p2_nm).reshape(-1, 1))
    
    

    matriz_frames_landmarks = reader.formar_matriz_frames_landmarks(landmarks_p1,landmarks_p2)

    matriz_frames_landmarks_nm=reader.formar_matriz_frames_landmarks(landmarks_p1_nm,landmarks_p2_nm)

    # Obtener las dimensiones de la matriz
    num_frames, _, num_landmarks, _ = matriz_frames_landmarks.shape
    # Crear una matriz flatten inicialmente vacía
    num_features = (num_landmarks * 4) + 6  # 2 personas * 2 coordenadas 
    matriz_flatten = np.zeros((num_frames, num_features))
    matriz_flatten_nm=np.zeros((num_frames, num_features))

    for i in range(num_frames):
        for j in range(num_landmarks):
            point1 = matriz_frames_landmarks[i, 0, j]
            point2 = matriz_frames_landmarks[i, 1, j]
            matriz_flatten[i, j * 4] = point1[0]  # x1
            matriz_flatten[i, j * 4 + 1] = point1[1]  # y1
            matriz_flatten[i, j * 4 + 2] = point2[0]  # x2
            matriz_flatten[i, j * 4 + 3] = point2[1]  # y2

            point1_nm = matriz_frames_landmarks_nm[i, 0, j]
            point2_nm = matriz_frames_landmarks_nm[i, 1, j]
            matriz_flatten_nm[i, j * 4] = point1_nm[0]  # x1
            matriz_flatten_nm[i, j * 4 + 1] = point1_nm[1]  # y1
            matriz_flatten_nm[i, j * 4 + 2] = point2_nm[0]  # x2
            matriz_flatten_nm[i, j * 4 + 3] = point2_nm[1]  # y2

         # Agregar los valores de yaw, pitch y roll después de terminar de recorrer los landmarks en el frame
        matriz_flatten[i, 196] = yaw_p1[i]
        matriz_flatten[i,197 ] = yaw_p2[i]
        matriz_flatten[i,198] = roll_p1[i]
        matriz_flatten[i, 199] = roll_p2[i]
        matriz_flatten[i, 200] = pitch_p1[i]
        matriz_flatten[i, 201] = pitch_p2[i]

        # Agregar los valores de yaw, pitch y roll para la matriz de no mimetismo después de terminar de recorrer los landmarks en el frame
        matriz_flatten_nm[i, 196] = yaw_p1_nm[i]
        matriz_flatten_nm[i, 197] = yaw_p2_nm[i]
        matriz_flatten_nm[i, 198] = roll_p1_nm[i]
        matriz_flatten_nm[i, 199] = roll_p2_nm[i]
        matriz_flatten_nm[i, 200] = pitch_p1_nm[i]
        matriz_flatten_nm[i, 201] = pitch_p2_nm[i]


    matrices_mimetismo.append(matriz_flatten)
    matrices_no_mimetismo.append(matriz_flatten_nm)


# Convertir las listas de matrices en una única matriz para cada caso
X_mimetismo = np.array(matrices_mimetismo)
X_no_mimetismo = np.array(matrices_no_mimetismo)

# Crear etiquetas para mimetismo y no mimetismo (1 para mimetismo, 0 para no mimetismo)
Y_mimetismo = np.ones((X_mimetismo.shape[0], 1))  
Y_no_mimetismo = np.zeros((X_no_mimetismo.shape[0], 1))  


# Concatenar las matrices de características
X_total = np.concatenate((X_mimetismo, X_no_mimetismo), axis=0)

# Concatenar las etiquetas
Y_total = np.concatenate((Y_mimetismo, Y_no_mimetismo), axis=0)

# Obtener el índice de permutación aleatoria
perm = np.random.permutation(len(X_total))
# Mezclar las matrices y etiquetas según el índice de permutación
X_total_shuffled = X_total[perm]
Y_total_shuffled = Y_total[perm]
# Definir el porcentaje de datos para entrenamiento (por ejemplo, 80%)
train_split = 0.8
# Calcular la cantidad de datos para entrenamiento
train_samples = int(train_split * len(X_total_shuffled))
# Dividir las matrices y etiquetas en entrenamiento y prueba
X_train = X_total_shuffled[:train_samples]
Y_train = Y_total_shuffled[:train_samples]
X_test = X_total_shuffled[train_samples:]
Y_test = Y_total_shuffled[train_samples:]

np.save('X_train_1.npy', X_train)
np.save('Y_train_1.npy', Y_train)
np.save('X_test_1.npy', X_test)
np.save('Y_test_1.npy', Y_test)


