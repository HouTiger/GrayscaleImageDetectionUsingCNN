import numpy as np 
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.models import Model
import pydot
import kt_utils
import keras.backend as K
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.models import load_model
import h5py
from PIL import Image

K.set_image_data_format('channels_last')

detect_model = load_model('edge_detec_model.h5')
test_images = kt_utils.load_test_images()

for i in range(test_images.shape[0]):
    test = test_images[i, ]
    test = np.expand_dims(test, axis=0)
    pred = detect_model.predict(test)
    # print(pred.shape)
    pred = pred.reshape((236, 236))
    pred = (pred > 0.5)
    pred = pred * 255
    # plt.imshow(pred, cmap='gray')
    # plt.show()
    plt.imsave('edge_detection_test/' + str(i) + '.png', pred, cmap='gray')

colored_img = cv2.imread('Airplane.jpg', 1)
print(colored_img.shape)
colored_img = colored_img.reshape((3, 512, 512))
colored_img = np.expand_dims(colored_img, axis = 3)
pred = detect_model.predict(colored_img)
pred = np.sum(pred, axis = 0)
pred = (pred > 0.5)
pred = pred * 255
pred = pred.reshape((508, 508))
print(pred.shape)
plt.imsave('airplane_edge.png', pred, cmap='gray')
"""
pred.reshape((508, 508))

pred = (pred > 0.5)

pred = pred * 255

plt.imsave('baboon_edge.png', pred, cmap='gray')
"""








