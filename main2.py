import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Ruta de la carpeta "train"
train_path = "a2/data/train"


# Función para redimensionar una imagen
def resizeImage(image_path, newSize=(100, 100)):  # TODO: Hay que ajustar el newsize
    img = Image.open(image_path).resize(newSize)
    return np.array(img).flatten()


# Listas para almacenar las rutas de las imagenes y sus respectivas etiquetas
images = []
labels = []

# Recorrer las carpetas en la carpeta "train"
for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)

    # Recorrer las imágenes en la carpeta actual
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)

        # Cargar y redimensionar la imagen
        img_array = resizeImage(image_path)

        # Añadir la matriz de la imagen y su etiqueta a las listas
        images.append(img_array)
        labels.append(folder)

# Crear un DataFrame con las imágenes y las etiquetas
dataFrame = pd.DataFrame({"Image": images, "Label": labels})

# Mostrar las primeras filas del DataFrame
print(dataFrame.head())

# Dividir los datos en conjuntos de train y test
X_train, X_test, y_train, y_test = train_test_split(dataFrame["Image"].tolist(), dataFrame['Label'].tolist(),
                                                    test_size=0.15, random_state=42)

### IMPORTANT!!! Hay que hacer resize de las imagenes si o si, si no da un error al tener algunas más pixeles que otras:
# ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (1200,) + inhomogeneous part.


# Crear y entrenar el modelo SVM
svm = SVC(C=1.0, kernel='linear')  # TODO: Probar con diferentes parámetros
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

# Calcular el accuracy del modelo entrenado
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
## Me sale un accuracy = 0.231111111... (puede tardar un rato)
