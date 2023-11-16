import numpy as np
import mahotas
from skimage.feature import hog, local_binary_pattern
from skimage.util import img_as_ubyte
from PIL import Image


class ImageProcessor:

    @staticmethod
    def resize_image(image_path, new_size):
        img = Image.open(image_path).convert("L")
        resized_img = img.resize(new_size)
        resized_img_array = np.array(resized_img)
        return resized_img_array

    @staticmethod
    def get_hog_features(image, orientations, pixels_per_cell):
        features, _ = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, visualize=True)
        return features.flatten()

    @staticmethod
    def get_glcm_features(img_array):
        normalized_img = img_as_ubyte(img_array)
        return mahotas.features.haralick(normalized_img.astype(np.uint8)).mean(axis=0)

    @staticmethod
    def get_lbp_features(img_array, points=8, radius=3):
        return local_binary_pattern(img_array, P=points, R=radius, method='uniform').flatten()

