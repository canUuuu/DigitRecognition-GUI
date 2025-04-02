from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import os

class CNNModel:
    def __init__(self, input_shape, model_path="MODEL/cnn_model.h5"):
        self.model_path = model_path
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
            self.is_load = True
        else:
            print("Creating a new model")
            self.model = self.build_model(input_shape)
            self.is_load = False

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))

        model.add(Dense(32))
        model.add(Activation("relu"))

        model.add(Dense(10))
        model.add(Activation("softmax"))

        return model

    def summary(self):
        self.model.summary()

    def compile(self, loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self, x_train, y_train, epochs=5, validation_split=0.3):
        self.model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split)

    def evaluate(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print("Test loss on test samples: ", test_loss)
        print("Validation accuracy: ", test_acc)

    def save(self):
        self.model.save(self.model_path)
        print(f"Model saved at {self.model_path}")
        self.is_load = True
