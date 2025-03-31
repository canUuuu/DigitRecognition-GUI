from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from utils import *
import os
import numpy as np




# 调用函数并获取 mnist 数据集
# C:\Users\29192\.keras\datasets\mnist.npz
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalization to [0-1]
x_train , x_test = imageNormalization(x_train, x_test)

checkMNISTdata(x_train, y_train, x_test, y_test)
# checkMATPLOTLIB(x_train[0], y_train[0])

model = Sequential()

# x_train.shape:[60000, 28, 28, channel = 1]
model.add(Conv2D(64, (3, 3), input_shape = x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))