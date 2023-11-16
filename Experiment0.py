import os
import pandas as pd
from image_processor import ImageProcessor
from svm_trainer import SVMTrainer


class Experiment0:

    def __init__(self, train_path):
        self.train_path = train_path

    def run_experiment(self):
        # Listas para almacenar las características y las etiquetas
        features = []
        labels = []

        # Recorrer las carpetas en la carpeta "train"
        for folder in os.listdir(self.train_path):
            folder_path = os.path.join(self.train_path, folder)

            # Recorrer las imágenes en la carpeta actual
            for image in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image)

                # Cargar, redimensionar la imagen y extraer características de intensidad
                img_array = ImageProcessor.resize_image(image_path, (100, 100)).flatten()

                # Añadir las características y su etiqueta a las listas
                features.append(img_array)
                labels.append(folder)

        # Crear un DataFrame con las características y las etiquetas
        df = pd.DataFrame({"Features": features, "Label": labels})

        # Mostrar las primeras filas del DataFrame
        print(df.head())

        SVMTrainer.svm_rbf(df, True)
        # 100x100 Accuracy: 0.3566666

        # TODO: times, F1
