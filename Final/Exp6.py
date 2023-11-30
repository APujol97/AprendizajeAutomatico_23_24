import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Ruta de la carpeta "train"
train_path = "data/train"


def resizeImage(image_path, newSize):
    img = Image.open(image_path).resize(newSize)
    img_array = np.array(img).flatten()

    return img_array


for elem in [100, 250, 400]:
    images_list = []
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
                img_array = resizeImage(imagen_path, (elem, elem))

                # Añadir las características HOG y su etiqueta a las listas
                images_list.append(img_array)
                labels.append(carpeta)


    # Crear un DataFrame con las características HOG y las etiquetas
    df_nat = pd.DataFrame({"Features": images_list, "Label": labels})

    print(df_nat.head())

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(df_nat["Features"].tolist(),
                                                                        df_nat["Label"].tolist(), test_size=0.2,
                                                                        random_state=42)


    # Crear un modelo SVM lineal
    modelo_svm = SVC(kernel='linear')

    # Entrenar el modelo
    modelo_svm.fit(X_train, y_train)

    # Predecir las etiquetas en el conjunto de prueba
    y_pred = modelo_svm.predict(X_test)

    # Calcular la precisión del modelo
    precision = accuracy_score(y_test, y_pred)

    # Imprimir la precisión
    print(f"Precisión del modelo SVM lineal con resize {elem}: {precision}")

    #caso con StandardScaler
    # Estandarizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Crear un modelo SVM lineal
    modelo_svm_scaled = SVC(kernel='linear')

    # Entrenar el modelo
    modelo_svm_scaled.fit(X_train_scaled, y_train)

    # Predecir las etiquetas en el conjunto de prueba
    y_pred = modelo_svm_scaled.predict(X_test_scaled)

    # Calcular la precisión del modelo
    precision_scaled = accuracy_score(y_test, y_pred)

    # Imprimir la precisión
    print(f"Precisión del modelo SVM lineal estandarizado con resize {elem}: {precision_scaled}")