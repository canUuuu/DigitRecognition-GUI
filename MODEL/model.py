import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
import os
import pandas as pd

class SaveEpochMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_log_path):
        super(SaveEpochMetricsCallback, self).__init__()
        self.model_log_path = model_log_path
        if os.path.exists(self.model_log_path):
            # If file exists, read the CSV to append data
            self.df = pd.read_csv(self.model_log_path)
        else:
            # If the file doesn't exist, initialize a new DataFrame
            self.df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"])

    def on_epoch_end(self, epoch, logs=None):
        # Get metrics at the end of each epoch
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        train_accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')

        # Create a new DataFrame for the new row
        new_row = pd.DataFrame({
            "epoch": [epoch + 1],  # Epochs are 0-indexed in TensorFlow/Keras
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "train_accuracy": [train_accuracy],
            "val_accuracy": [val_accuracy]
        })

        # Use pd.concat to append the new row to the existing DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        # Save the updated DataFrame back to CSV
        self.df.to_csv(self.model_log_path, index=False)
class CNNModel:
    def __init__(self, input_shape, model_path="MODEL/cnn_model.h5"):
        self.model_path = model_path
        self.model_log_path = "result/epoch_loss_summary.csv"
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
            self.is_load = True
        else:
            print("Creating a new model")
            self.build_model(input_shape)
            self.is_load = False

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))
        # model.add(Conv2D(64, (3, 3)))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))

        # model.add(Dense(32))
        # model.add(Activation("relu"))

        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        self.model = model
        self.compile()

    def summary(self):
        self.model.summary()

    def compile(self, loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self, x_train, y_train, epochs=5, validation_split=0.3):
        save_callback = SaveEpochMetricsCallback(self.model_log_path)
        self.model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[save_callback])

    def evaluate(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print("Test loss on test samples: ", test_loss)
        print("Validation accuracy: ", test_acc)

    def save(self):
        self.model.save(self.model_path)
        print(f"Model saved at {self.model_path}")
        self.is_load = True

    def predict_digit(self, img):
        test_image = img.reshape(-1, 28, 28, 1)
        # return np.argmax(self.model.predict(test_image))
        return self.model.predict(test_image)