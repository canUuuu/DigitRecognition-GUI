from MODEL.model import CNNModel
from utils import *

# getting dataset
# C:\Users\29192\.keras\datasets\mnist.npz
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalization to [0-1]
x_train , x_test = imageNormalization(x_train, x_test)

checkMNISTdata(x_train, y_train, x_test, y_test)
# checkMATPLOTLIB(x_train[0], y_train[0])

# ===================== model =====================
model_path = "MODEL/cnn_model.h5"
model = CNNModel(x_train.shape[1:], model_path = model_path)
model.summary()
# train
if not model.is_load:
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics=['accuracy'])
    model.train(x_train, y_train, epochs = 5, validation_split = 0.3)
    # save
    model.save()
# evaluate
model.evaluate()
# ===================== model =====================
