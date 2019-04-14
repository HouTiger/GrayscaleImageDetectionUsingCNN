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
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

K.set_image_data_format('channels_last')
detect_model = load_model('cor_detec_model.h5')
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
    plt.imsave('corner_detection_test/' + str(i) + '.png', pred, cmap='gray')


plot_model(detect_model, to_file='cor_detect_model.png')
SVG(model_to_dot(detect_model).create(prog='dot', format='svg'))





