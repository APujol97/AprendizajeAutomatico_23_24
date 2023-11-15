import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from PIL import Image

initial_time = time.time()

# Función para cargar y redimensionar una imagen
def resizeImage(image_path, newSize=(400, 400)):
    img = Image.open(image_path).resize(newSize, Image.NEAREST) # NEAREST indica que se hace el resize sin crear un marco blanco o negro, se aumenta la imagen
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

    # Flattening para obtener un array 1D
    features_flattened = features.flatten()

    return features_flattened


# Ruta de la carpeta "train"
train_path = "a2/data/train"

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
            img_array = resizeImage(imagen_path)

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

# Estandarizar los datos
scaler = StandardScaler()
X_train_hog_scaled = scaler.fit_transform(X_train_hog)
X_test_hog_scaled = scaler.transform(X_test_hog)

# Crear un modelo SVM lineal
modelo_svm_hog = SVC()

# Definir los parámetros a buscar en la búsqueda exhaustiva

parametros_grid_linear = {
    'kernel': ['linear'],
    'C': [0.1, 1, 10, 100],
}

parametros_grid_rbf = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
}

parametros_grid_poly = {
    'kernel': ['poly'],
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],
    'coef0': [0.0, 1.0, 2.0],
}

parametros_grid_sigmoid = {
    'kernel': ['sigmoid'],
    'C': [0.1, 1, 10, 100],
    'coef0': [0.0, 1.0, 2.0],
    'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
}

parametros_grid = [parametros_grid_linear, parametros_grid_rbf, parametros_grid_poly, parametros_grid_sigmoid]

# Crear un objeto KFold para la validación cruzada
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(
    estimator=modelo_svm_hog,
    param_grid=parametros_grid,
    scoring='accuracy',
    cv=kfold,
    n_jobs=-1,
    verbose=2
)

# Entrenar el modelo con búsqueda exhaustiva y validación cruzada
grid_search.fit(X_train_hog_scaled, y_train_hog)

# Obtener el mejor modelo y sus parámetros
mejor_modelo = grid_search.best_estimator_
mejores_parametros = grid_search.best_params_

# Predecir las etiquetas en el conjunto de prueba usando el mejor modelo
y_pred_hog = mejor_modelo.predict(X_test_hog_scaled)

# Calcular la precisión del modelo
precision_hog = accuracy_score(y_test_hog, y_pred_hog)

# Imprimir los mejores parámetros y la precisión
print(f"Mejores parámetros: {mejores_parametros}")
print(f"Precisión del modelo SVM con características HOG: {precision_hog}")

final_time = time.time()

execution_time = final_time - initial_time

print(f"Tiempo de ejecución: {execution_time} segundos")

# Mejores parámetros: {'C': 0.1, 'gamma': 1, 'kernel': 'linear'}
# Precisión del modelo SVM con características HOG: 0.5766666666666667

# Mejores parámetros: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
# Precisión del modelo SVM con características HOG: 0.5633333333333334

# Mejores parámetros: {'C': 10, 'coef0': 0.0, 'gamma': 'scale', 'kernel': 'sigmoid'} (150x150) sin escala grises
# Precisión del modelo SVM con características HOG: 0.5933333333333334
# Tiempo de ejecución: 4207.495000123978 segundos (1h 10m 8s)

# Mejores parámetros: {'C': 10, 'coef0': 1.0, 'gamma': 'auto', 'kernel': 'sigmoid'} (400x400) con escala de grises
# Precisión del modelo SVM con características HOG: 0.66
# Tiempo de ejecución: 39854.49985194206 segundos (11h 4m 14s)