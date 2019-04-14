import numpy as np
from keras.preprocessing import image
import cv2
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from matplotlib.pyplot import imshow
def load_data_edge():
    data = np.array(np.load("train_data.npy"))
    data = data[0:100390 + 110910, :]
    np.random.shuffle(data)
    print(data.shape)
    X = data[:, 0:25]
    Y = data[:, 25]
    X = X.reshape((X.shape[0], 5, 5, 1))
    Y = Y.reshape((Y.shape[0], 1, 1, 1))
    print(X.shape)
    print(Y.shape)
    size_of_train_set = int(X.shape[0] * 0.7)
    X_train = X[0:size_of_train_set, :]
    X_test = X[size_of_train_set: , :]
    Y_train = Y[0:size_of_train_set, :]
    Y_test = Y[size_of_train_set:, :]
    print("X_train.shape = ", X_train.shape)
    print("X_test.shape = ", X_test.shape)
    print("Y_train.shape = ", Y_train.shape)
    print("Y_test.shape = ", Y_test.shape)
    print(X_train[0, :])
    return X_train, Y_train, X_test, Y_test

def load_data_corner():
    data = np.load("train_data.npy")
    print(data.shape)
    data[0:211300, 25] = 0
    data[211300:, 25] = 1
    np.random.shuffle(data)
    X = data[:, 0:25]
    Y = data[:, 25]
    print(X.shape)
    print(Y.shape)
    X = X.reshape((X.shape[0], 5, 5, 1))
    Y = Y.reshape((Y.shape[0], 1, 1, 1))
    print(X.shape)
    print(Y.shape)
    size_of_train_set = int(X.shape[0] * 0.7)
    X_train = X[0:size_of_train_set, :]
    X_test = X[size_of_train_set:, :]
    Y_train = Y[0:size_of_train_set, :]
    Y_test = Y[size_of_train_set:, :]
    print("X_train.shape = ", X_train.shape)
    print("X_test.shape = ", X_test.shape)
    print("Y_train.shape = ", Y_train.shape)
    print("Y_test.shape = ", Y_test.shape)
    # print(X_train[100])
    return X_train, Y_train, X_test, Y_test




def load_test_images():
    data = []

    pre_path = 'synthetic_characters/'
    for i in range(4):
        for j in range(10):
            image_path = pre_path + '0-0-0-' + str(i) + '-' + str(j) + '.bmp'
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # print(type(img))
            # print(img.shape)
            data.append(img)

    data = np.array(data)
    data = np.expand_dims(data, axis=3)
    print('test images dim = ', data.shape)
    return data