import os
import pandas as pd
import numpy as np
import time
from image_processor import ImageProcessor
from svm_trainer import SVMTrainer


class Exp8:

    def __init__(self, train_path):
        self.train_path = train_path

    def run_experiment(self):
        initial_time = time.time()

        # Listas para almacenar las características HOG y sus respectivas etiquetas
        features = []
        labels = []

        # Recorrer las carpetas en la carpeta "train"
        for carpeta in os.listdir(self.train_path):
            folder_path = os.path.join(self.train_path, carpeta)

            # Verificar si es una carpeta
            if os.path.isdir(folder_path):

                # Recorrer las imágenes en la carpeta actual
                for image in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image)

                    # Cargar y redimensionar la imagen a escala de grises
                    img_array = ImageProcessor.resize_image(image_path, (200, 200))

                    # Calcular las características HOG de la imagen
                    hog_features = ImageProcessor.get_hog_features(img_array, 10, (8, 8))

                    # Añadir las características HOG y su etiqueta a las listas
                    features.append(hog_features)
                    labels.append(carpeta)

        # Crear un DataFrame con las características HOG y las etiquetas
        df = pd.DataFrame({"Features": features, "Label": labels})

        print(df.head())

        SVMTrainer.svm_gridSearch(df, initial_time)

