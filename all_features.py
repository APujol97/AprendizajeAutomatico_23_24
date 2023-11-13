import os
import pandas as pd
import numpy as np
import mahotas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIL import Image


# Función para cargar, redimensionar una imagen y convertirla en escala de grises si no lo está
def resizeImage(image_path, newSize=(100, 100)):
    img = Image.open(image_path).resize(newSize)
    img_array = np.array(img).flatten()

    if len(img_array.shape) == 3:
        # Si la imagen es a color, convertir a escala de grises y luego extraer características
        gray_img = np.array(img.convert("L"))
    else:
        gray_img = np.array(img)

    return gray_img


# Función para calcular las características HOG de una imagen
def GetHogFeatures(imagen):
    # Calcular las características HOG
    features, _ = hog(imagen, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)

    return features.flatten()

# Función para calcular las características GLCM de una imagen
def GetGlcmFeatures(img_array):
    return mahotas.features.haralick(img_array.astype(np.uint8)).mean(axis=0)

# Función para calcular las características LBP de una imagen
def GetLbpFeatures(img_array, points=8, radius=3):
    return local_binary_pattern(img_array, P=points, R=radius, method='uniform').flatten()

# Ruta de la carpeta "train"
train_path = "a2/data/train"

# Listas para almacenar las características HOG y sus respectivas etiquetas
allFeatures = []
labels = []

# Recorrer las carpetas en la carpeta "train"
for carpeta in os.listdir(train_path):
    folder_path = os.path.join(train_path, carpeta)

    # Verificar si es una carpeta
    if os.path.isdir(folder_path):

        # Recorrer las imágenes en la carpeta actual
        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image)

            # Cargar y redimensionar la imagen a escala de grises
            img_array = resizeImage(image_path)

            # Calcular las características HOG de la imagen
            hog_features = GetHogFeatures(img_array)

            # Calcular las características GLCM de la imagen
            glcm_features = GetGlcmFeatures(img_array)

            # Calcular las características LBP de la imagen
            lbp_features = GetLbpFeatures(img_array)

            # Añadir las características HOG y su etiqueta a las listas
            allFeatures.append(np.concatenate((hog_features, glcm_features, lbp_features)))
            labels.append(carpeta)

# Crear un DataFrame con las características HOG y las etiquetas
df = pd.DataFrame({"Features": allFeatures, "Label": labels})

print(df.head())
# Dividir los datos en conjuntos de entrenamiento y prueba
#df_hog = shuffle(df_hog, random_state=42)  # Mezclar los datos
X_train, X_test, y_train, y_test = train_test_split(df["Features"].tolist(),
                                                                    df["Label"].tolist(), test_size=0.2,
                                                                    random_state=42)

# Estandarizar los datos
scaler = MinMaxScaler() #StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear un modelo SVM lineal
svm = SVC(kernel='linear')

# Entrenar el modelo
svm.fit(X_train_scaled, y_train)
#svm.fit(X_train, y_train)

# Predecir las etiquetas en el conjunto de prueba
y_pred = svm.predict(X_test_scaled)
#y_pred = svm.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Imprimir la precisión
print(f"Precisión del modelo SVM lineal: {accuracy}")

# HOG + GCLM + LBP
# StandardScaler
# Accuracy: 0.553333333

# HOG + GCLM + LBP
# StandardScaler
# Accuracy: 0.53

# HOG + GLCM + LBP
# Sin estandarizar
# Accuracy: 0.353333333

