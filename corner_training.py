import numpy as np
from keras import layers
from keras.layers import Input, MaxPooling2D, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import kt_utils
import keras.backend as K
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import h5py
K.set_image_data_format('channels_last')

X_train_orig, Y_train, X_test_orig, Y_test = kt_utils.load_data_corner()
X_train = X_train_orig / 255
X_test = X_test_orig / 255


def corner_detect_model(input_shape):
    X_input = Input(input_shape)

    # (5, 5, 1) -> (3, 3, 8), filter = (3, 3)
    X = Conv2D(8, (3, 3), strides=(1, 1), name='conv0')(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # (3, 3, 8) -> (3, 3, 8)
    X = Conv2D(8, (1, 1), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)

    # (3, 3, 8) -> (1, 1, 16)
    X = Conv2D(16, (3, 3), strides=(1, 1), name='conv2')(X)
    X = Activation('relu')(X)

    # (1, 1, 16) -> (1, 1, 3)
    X = Conv2D(3, (1, 1), strides=(1, 1), name='conv3')(X)
    X = Activation('relu')(X)

    # (1, 1, 3) -> (1, 1, 1)
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv4')(X)
    X = Activation('sigmoid')(X)
    model = Model(inputs=X_input, outputs=X, name='corner_detect_model')
    return model


cor_detect_model = corner_detect_model((None, None, 1))
cor_detect_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
cor_detect_model.fit(x=X_train, y=Y_train, epochs=10, batch_size=16)
preds = cor_detect_model.evaluate(x=X_test, y=Y_test)

print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
cor_detect_model.save('cor_detec_model.h5')