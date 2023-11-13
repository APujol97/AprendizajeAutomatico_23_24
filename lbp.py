import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from skimage import color

# Ruta de la carpeta "train"
train_path = "a2/data/train"


# Función para redimensionar una imagen y extraer características de LBP (Local Binary Pattern)
def resizeImage(image_path, newSize=(100, 100), lbp_radius=3, lbp_points=8):    # TODO: cambiar radio y puntos¿?¿?
    img = Image.open(image_path).resize(newSize)

    # Verificar si la imagen es en escala de grises
    if img.mode == "RGB":
        # Convertir la imagen a escala de grises
        gray_img = color.rgb2gray(np.array(img))
    else:
        gray_img = np.array(img)

    # Calcular LBP
    lbp = local_binary_pattern(gray_img, P=lbp_points, R=lbp_radius, method='uniform')

    return lbp.flatten()


# Listas para almacenar las características y las etiquetas
features = []
labels = []

# Recorrer las carpetas en la carpeta "train"
for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)

    # Recorrer las imágenes en la carpeta actual
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)

        # Cargar, redimensionar la imagen y extraer características de LBP
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

# Accuracy: 0.23555555 con lbp_radius=3, lbp_points=8

