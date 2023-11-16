import os
import pandas as pd
import numpy as np
from image_processor import ImageProcessor
from svm_trainer import SVMTrainer


class Experiment3:

    def __init__(self, train_path):
        self.train_path = train_path

    def run_experiment(self):

        # Listas para almacenar las características HOG y sus respectivas etiquetas
        allFeatures = []
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
                    img_array = ImageProcessor.resize_image(image_path, (100, 100))  # TODO: ajustar if needed

                    # Calcular las características HOG de la imagen
                    hog_features = ImageProcessor.get_hog_features(img_array, 8, (4, 4))  # TODO: ajustar

                    # Calcular las características GLCM de la imagen
                    glcm_features = ImageProcessor.get_glcm_features(img_array)

                    # Calcular las características LBP de la imagen
                    lbp_features = ImageProcessor.get_lbp_features(img_array)

                    # Añadir las características HOG y su etiqueta a las listas
                    allFeatures.append(np.concatenate((hog_features, glcm_features, lbp_features)))
                    labels.append(carpeta)

        # Crear un DataFrame con las características HOG y las etiquetas
        df = pd.DataFrame({"Features": allFeatures, "Label": labels})

        print(df.head())

        SVMTrainer.svm_rbf(df, plotReport=True)

        # HOG + GCLM + LBP
        # StandardScaler
        # Accuracy: 0.553333333

        # HOG + GCLM + LBP
        # MinMaxScaler
        # Accuracy: 0.53

        # HOG + GLCM + LBP
        # Sin estandarizar
        # Accuracy: 0.353333333
