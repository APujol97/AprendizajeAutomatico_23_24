import time
import os
import numpy as np
from svm_trainer import SVMTrainer
from image_processor import ImageProcessor
from sklearn.preprocessing import LabelEncoder


class Test:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.trainer = SVMTrainer()

    def load_data(self, folder_path, label_names, label_encoder=None):
        all_features = []
        labels = []

        if label_encoder is None:
            label_encoder = LabelEncoder()

        for folder in os.listdir(folder_path):
            class_folder_path = os.path.join(folder_path, folder)
            folder = folder.lower()
            if folder in label_names:
                if os.path.isdir(class_folder_path):
                    for image_name in os.listdir(class_folder_path):
                        image_path = os.path.join(class_folder_path, image_name)
                        img_array = ImageProcessor.resize_image(image_path, (200, 200))

                        hog_features = ImageProcessor.get_hog_features(img_array, 10, (8, 8))
                        glcm_features = ImageProcessor.get_glcm_features(img_array)
                        lbp_features = ImageProcessor.get_lbp_features(img_array)

                        all_features.append(np.concatenate((hog_features, glcm_features, lbp_features)))
                        labels.append(folder)

        if label_encoder is None or not hasattr(label_encoder, 'classes_'):
            label_encoder.fit(np.unique(labels))

        encoded_labels = label_encoder.transform(labels)

        return np.array(all_features), np.array(encoded_labels), label_encoder

    def run_experiment(self):

        folder_names = []

        for item in os.listdir(self.train_path):
            item_path = os.path.join(self.train_path, item)

            if os.path.isdir(item_path):
                folder_names.append(item.lower())


        # Cargar datos de entrenamiento
        X_train, y_train, label_encoder = self.load_data(self.train_path, label_names=folder_names)

        # Iniciar tiempo de ejecución
        start_time = time.time()


        # Entrenar el modelo
        self.trainer.train(X_train, y_train)
        y_train_pred = self.trainer.predict(X_train)
        self.trainer.evaluate(X_train, y_train)

        # Cargar datos de test
        X_test, y_test, _ = self.load_data(self.test_path, label_encoder=label_encoder, label_names=folder_names)

        # Hacer predicciones en datos de prueba
        y_pred = self.trainer.predict(X_test)

        # Evaluar en datos de prueba
        self.trainer.evaluate(X_test, y_test)

        # Calcular y mostrar el tiempo de ejecución total
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución total: {execution_time} segundos")