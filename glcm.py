import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mahotas as mh

# Ruta de la carpeta "train"
train_path = "a2/data/train"

# Función para redimensionar una imagen y extraer características de textura (Gray-Level Co-Occurrence)
def resizeImage(image_path, newSize=(100, 100)):
    img = Image.open(image_path).resize(newSize)
    img_array = np.array(img).flatten()

    # Verificar si la imagen es en escala de grises
    if len(img_array.shape) == 2:
        # Si la imagen es en escala de grises, no se realiza ninguna conversión
        texture_features = mh.features.haralick(img_array.astype(np.uint8)).mean(axis=0)
    else:
        # Si la imagen es a color, convertir a escala de grises y luego extraer características
        gray_img = np.array(img.convert("L"))
        texture_features = mh.features.haralick(gray_img.astype(np.uint8)).mean(axis=0)

    # Añadir características de textura al array de la imagen
    img_array = np.concatenate((img_array, texture_features))

    return img_array

# Listas para almacenar las características y las etiquetas
features = []
labels = []

# Recorrer las carpetas en la carpeta "train"
for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)

    # Recorrer las imágenes en la carpeta actual
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)

        # Cargar, redimensionar la imagen y extraer características de textura
        img_array = resizeImage(image_path)

        # Añadir las características y su etiqueta a las listas
        features.append(img_array)
        labels.append(folder)

# Crear un DataFrame con las características y las etiquetas
dataFrame = pd.DataFrame({"Features": features, "Label": labels})

# Mostrar las primeras filas del DataFrame
print(dataFrame.head())

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(dataFrame["Features"].tolist(), dataFrame['Label'].tolist(),
                                                    test_size=0.15, random_state=42)

# Crear y entrenar el modelo SVM
svm = SVC(C=1.0, kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

# Calcular la precisión del modelo entrenado
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Me da un accuracy: 0.29333333333333