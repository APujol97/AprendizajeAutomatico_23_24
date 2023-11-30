import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from PIL import Image
import numpy as np

train_path = "a2/data/train"


def resize_image(imagen_path):
    img = Image.open(imagen_path)
    img_new = Image.new(img.mode, (411, 411), 255)
    img_new.paste(img, img.getbbox())
    return np.array(img_new)


def GetHogFeatures(imagen):
    # Calcular las características HOG
    features, _ = hog(imagen, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)

    # Flattening para obtener un array 1D
    features_flattened = features.flatten()

    return features_flattened


# Listas para almacenar las características HOG y sus respectivas etiquetas
features = []
labels = []

# Recorrer las carpetas en la carpeta "train"
for carpeta in os.listdir(train_path):
    carpeta_path = os.path.join(train_path, carpeta)

    # Verificar si es una carpeta
    if os.path.isdir(carpeta_path):

        # Recorrer las imágenes en la carpeta actual
        for imagen in os.listdir(carpeta_path):
            imagen_path = os.path.join(carpeta_path, imagen)

            # Cargar y redimensionar la imagen
            img_array = resize_image(imagen_path)

            # Calcular las características HOG de la imagen
            hog_features = GetHogFeatures(img_array)

            # Añadir las características HOG y su etiqueta a las listas
            features.append(hog_features)
            labels.append(carpeta)

# Crear un DataFrame con las características HOG y las etiquetas
df_hog = pd.DataFrame({"Features": features, "Label": labels})

print(df_hog.head())
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_hog, X_test_hog, y_train_hog, y_test_hog = train_test_split(df_hog["Features"].tolist(),
                                                                    df_hog["Label"].tolist(), test_size=0.2,
                                                                    random_state=42)

# Crear un modelo SVM lineal
modelo_svm_hog = SVC(kernel='linear')

# Entrenar el modelo
modelo_svm_hog.fit(X_train_hog, y_train_hog)

# Predecir las etiquetas en el conjunto de prueba
y_pred_hog = modelo_svm_hog.predict(X_test_hog)

# Calcular la precisión del modelo
precision_hog = accuracy_score(y_test_hog, y_pred_hog)

# Imprimir la precisión
print(f"Precisión del modelo SVM lineal con características HOG: {precision_hog}")
