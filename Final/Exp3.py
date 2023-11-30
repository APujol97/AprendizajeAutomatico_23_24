import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mahotas
from PIL import Image
import numpy as np

train_path = "a2/data/train"


def resize_image(image_path, new_size):
    img = Image.open(image_path).convert("L")
    resized_img = img.resize(new_size)
    resized_img_array = np.array(resized_img)
    return resized_img_array


# Función para calcular las características GLCM de una imagen
def GetGlcmFeatures(img_array):
    return mahotas.features.haralick(img_array.astype(np.uint8)).mean(axis=0)


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
            glcm_features = GetGlcmFeatures(img_array)

            # Añadir las características HOG y su etiqueta a las listas
            features.append(glcm_features)
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
