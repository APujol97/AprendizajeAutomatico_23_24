import os
import pandas as pd
import numpy as np
import time
from image_processor import ImageProcessor
from svm_trainer import SVMTrainer


class Experiment4:

    def __init__(self, train_path):
        self.train_path = train_path

    def run_experiment(self):
        all_features = []
        labels = []

        initial_time = time.time()

        for folder in os.listdir(self.train_path):
            folder_path = os.path.join(self.train_path, folder)

            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    img_array = ImageProcessor.resize_image(image_path, (200, 200))

                    hog_features = ImageProcessor.get_hog_features(img_array, 8, (4, 4))  # TODO: change to optimal
                    glcm_features = ImageProcessor.get_glcm_features(img_array)
                    lbp_features = ImageProcessor.get_lbp_features(img_array)

                    all_features.append(np.concatenate((hog_features, glcm_features, lbp_features)))
                    labels.append(folder)

        df = pd.DataFrame({"Features": all_features, "Label": labels})

        best_precision = SVMTrainer.svm_gridSearch(df, initial_time)

        print(f"\nPrecisión del modelo final: {best_precision}")
