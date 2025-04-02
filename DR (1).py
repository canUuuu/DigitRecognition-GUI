from MODEL.model import cnnModel
from utils import *
from tensorflow.keras.models import Sequential

import os
import numpy as np

# getting dataset
# C:\Users\29192\.keras\datasets\mnist.npz
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalization to [0-1]
x_train , x_test = imageNormalization(x_train, x_test)

checkMNISTdata(x_train, y_train, x_test, y_test)
# checkMATPLOTLIB(x_train[0], y_train[0])

model = cnnModel(x_train.shape[1:])
model.summary()


# train

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 5, validation_split = 0.3)

model.sava("MODEL/cnnModel.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss on test samples: ", test_loss)
print("Validation accuracy: ", test_acc)
