import os
import pandas as pd
import time
from image_processor import ImageProcessor
from svm_trainer import SVMTrainer


class Exp7:

    def __init__(self, train_path):
        self.train_path = train_path

    def run_experiment(self):
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
                    for carpeta in os.listdir(self.train_path):
                        carpeta_path = os.path.join(self.train_path, carpeta)

                        # Verifica si es una carpeta
                        if os.path.isdir(carpeta_path):
                            # Recorre las imágenes en la carpeta actual
                            for imagen in os.listdir(carpeta_path):
                                imagen_path = os.path.join(carpeta_path, imagen)

                                # Carga y redimensiona la imagen
                                img_array = ImageProcessor.resize_image(imagen_path, newSize)

                                # Calcula las características HOG de la imagen
                                hog_features = ImageProcessor.get_hog_features(img_array, orientations, pixels_per_cell)

                                # Añade las características HOG y su etiqueta a las listas
                                features.append(hog_features)
                                labels.append(carpeta)

                    # Crea un DataFrame con las características HOG y las etiquetas
                    df = pd.DataFrame({"Features": features, "Label": labels})

                    # Calcula la precisión del modelo
                    accuracy = SVMTrainer.svm_rbf(df, plotReport=False)

                    final_exec_time = time.time()
                    exec_time = final_exec_time - initial_time

                    # Imprime la precisión y los parámetros utilizados
                    print(f"Precisión del modelo SVM lineal con características HOG (newSize={newSize}, "
                          f"orientations={orientations}, pixels_per_cell={pixels_per_cell}): {accuracy}, ExecTime: {exec_time}")

                    # Guarda los mejores parámetros si la precisión actual está entre los 5 mejores
                    top_params.append({
                        'newSize': newSize,
                        'orientations': orientations,
                        'pixels_per_cell': pixels_per_cell,
                        'accuracy': accuracy,
                        'time': exec_time
                    })

        # Ordena la lista de mejores parámetros por precisión en orden descendente
        top_params = sorted(top_params, key=lambda x: x['accuracy'], reverse=True)[:5]

        print("Top 5 mejores parámetros:")
        for i, params in enumerate(top_params, 1):
            print(f"{i}. Parámetros: {params}")

        final_time = time.time()

        print(f"Tiempo total de ejecución: {final_time - init_time}")
