import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

from skimage.feature import hog
from skimage import data, exposure
import skimage
import os
import pandas as pd

filename = "./data/train/bedroom/image_0001.jpg"
moon = skimage.io.imread(filename)

print(moon)

list_folders_names = []
list_folders_names.append("./data/train/bedroom/")
list_folders_names.append("./data/train/Coast/")
list_folders_names.append("./data/train/Forest/")
list_folders_names.append("./data/train/Highway/")
list_folders_names.append("./data/train/industrial/")
list_folders_names.append("./data/train/Insidecity/")
list_folders_names.append("./data/train/kitchen/")
list_folders_names.append("./data/train/livingroom/")
list_folders_names.append("./data/train/Mountain/")
list_folders_names.append("./data/train/Office/")
list_folders_names.append("./data/train/OpenCountry/")
list_folders_names.append("./data/train/store/")
list_folders_names.append("./data/train/Street/")
list_folders_names.append("./data/train/Suburb/")
list_folders_names.append("./data/train/TallBuilding/")


list_folders = []
for elem in list_folders_names:
    list_folders.append(os.listdir(elem))

image_list = []
for folder, folder_name in zip(list_folders, list_folders_names):
    for filename in folder:
        image_list.append(skimage.io.imread(folder_name+filename))

#hog_image_list = []
hog_image_vector_list = []
for elem in image_list:
    fd, hog_image_vector = hog(
        elem,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        feature_vector=True,
    )
    #hog_image_list.append(hog_image)
    hog_image_vector_list.append(hog_image_vector)

np.save("matrix_features_vector", hog_image_vector_list)
#dataframe = pd.DataFrame(hog_image_list)

#print(dataframe.head())

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

#ax1.axis('Off')
#ax1.imshow(image_list[101], cmap=plt.get_cmap('gray'))
#ax1.set_title('Input image')

#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

#ax2.axis('off')
#ax2.imshow(hog_image_rescaled, cmap=plt.get_cmap('gray'))
#ax2.set_title('Histogram of Oriented Grafients')
#plt.show()

