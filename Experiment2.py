import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.model_selection import GridSearchCV


# Función para cargar, redimensionar una imagen y convertirla en escala de grises si no lo está
def resizeImage(image_path, newSize):
    img = Image.open(image_path).resize(newSize, Image.NEAREST)
    img_array = np.array(img).flatten()

    if len(img_array.shape) == 3:
        # Si la imagen es a color, convertir a escala de grises y luego extraer características
        gray_img = np.array(img.convert("L"))
    else:
        gray_img = np.array(img)

    return gray_img

# Función para calcular las características HOG de una imagen
def GetHogFeatures(imagen, orientations, pixels_per_cell):
    # Calcular las características HOG
    featuresHog= hog(imagen,
                      orientations=orientations,
                      pixels_per_cell=pixels_per_cell,
                      visualize=False)

    # Flattening para obtener un array 1D
    features_flattened = featuresHog.flatten()

    return features_flattened


# Ruta de la carpeta "train"
train_path = "a2/data/train"

# Listas para almacenar las características HOG y sus respectivas etiquetas
features = []
labels = []

# Define los parámetros a optimizar
param_grid = {
    'orientations': [8, 9, 10],
    'pixels_per_cell': [(4, 4), (6, 6), (8, 8)],
    'newSize': [(50, 50), (100, 100), (200, 200), (250, 250)]
}

init_time = time.time()

top_params = []

# Recorre las combinaciones de parámetros
for newSize in param_grid['newSize']:
    for orientations in param_grid['orientations']:
        for pixels_per_cell in param_grid['pixels_per_cell']:
            # Lista para almacenar las características HOG y sus respectivas etiquetas
            features = []
            labels = []

            initial_time = time.time()

            # Recorre las carpetas en la carpeta "train"
            for carpeta in os.listdir(train_path):
                carpeta_path = os.path.join(train_path, carpeta)

                # Verifica si es una carpeta
                if os.path.isdir(carpeta_path):
                    # Recorre las imágenes en la carpeta actual
                    for imagen in os.listdir(carpeta_path):
                        imagen_path = os.path.join(carpeta_path, imagen)

                        # Carga y redimensiona la imagen
                        img_array = resizeImage(imagen_path, newSize)

                        # Calcula las características HOG de la imagen
                        hog_features = GetHogFeatures(img_array, orientations, pixels_per_cell)

                        # Añade las características HOG y su etiqueta a las listas
                        features.append(hog_features)
                        labels.append(carpeta)

            # Crea un DataFrame con las características HOG y las etiquetas
            df_hog = pd.DataFrame({"Features": features, "Label": labels})

            # Divide los datos en conjuntos de entrenamiento y prueba
            X_train_hog, X_test_hog, y_train_hog, y_test_hog = train_test_split(df_hog["Features"].tolist(),
                                                                                df_hog["Label"].tolist(), test_size=0.2,
                                                                                random_state=42)

            # Estandariza los datos
            scaler = StandardScaler()
            X_train_hog_scaled = scaler.fit_transform(X_train_hog)
            X_test_hog_scaled = scaler.transform(X_test_hog)

            # Entrenar el modelo SVM lineal
            modelo_svm_hog = SVC(kernel='rbf')

            # Entrenar el modelo
            modelo_svm_hog.fit(X_train_hog_scaled, y_train_hog)

            # Predice las etiquetas en el conjunto de prueba
            y_pred_hog = modelo_svm_hog.predict(X_test_hog_scaled)

            # Calcula la precisión del modelo
            precision_hog = accuracy_score(y_test_hog, y_pred_hog)

            final_exec_time = time.time()
            exec_time = final_exec_time - initial_time
            # Imprime la precisión y los parámetros utilizados
            print(f"Precisión del modelo SVM lineal con características HOG (newSize={newSize}, "
                  f"orientations={orientations}, pixels_per_cell={pixels_per_cell}): {precision_hog}, ExecTime: {exec_time}")

            # Guarda los mejores parámetros si la precisión actual está entre los 5 mejores
            top_params.append({
                'newSize': newSize,
                'orientations': orientations,
                'pixels_per_cell': pixels_per_cell,
                'precision': precision_hog,
                'time': exec_time
            })

# Ordena la lista de mejores parámetros por precisión en orden descendente
top_params = sorted(top_params, key=lambda x: x['precision'], reverse=True)[:5]

print("Top 5 mejores parámetros:")
for i, params in enumerate(top_params, 1):
    print(f"{i}. Parámetros: {params}")

final_time = time.time()


print(f"Tiempo total de ejecución: {final_time-init_time}")