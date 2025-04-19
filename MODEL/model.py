import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
from cpaslayer import CapsLayer
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

class BaseModel(ABC):
    label = "base_model"
    def __init__(self, input_shape, model_path=None):
        if model_path is None:
            model_path = f"MODEL/{self.label}.h5"
        self.model_path = model_path
        self.model_log_path = f"result/{self.label}_epoch_loss_summary.csv"
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
            self.is_load = True
        else:
            print("Creating a new model")
            self.build_model(input_shape)
            self.is_load = False

    @abstractmethod
    def build_model(self, input_shape):
        """子类需要实现具体模型结构"""
        pass

    def compile(self, loss="sparse_categorical_crossentropy", learning_rate=0.001, metrics=["accuracy"]):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def summary(self):
        self.model.summary()

    def train(self, x_train, y_train, epochs=5, validation_split=0, batch_size=32):
        save_callback = SaveEpochMetricsCallback(self.model_log_path)
        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
            callbacks=[save_callback]
        )

    def evaluate(self, x_test, y_test, n_splits=10, save_path="result/acc.csv"):
        batch_size = len(x_test) // n_splits
        acc_list = []

        for i in range(n_splits):
            start = i * batch_size
            end = (i + 1) * batch_size if i != n_splits - 1 else len(x_test)

            x_batch = x_test[start:end]
            y_batch = y_test[start:end]

            y_pred = self.model.predict(x_batch)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_batch, axis=1) if y_batch.ndim > 1 else y_batch

            acc = accuracy_score(y_true_labels, y_pred_labels)
            acc_list.append(acc)

        print("Accuracy for each partition:", acc_list)
        print("Average accuracy:", np.mean(acc_list))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame({
            "Partition": np.arange(1, n_splits + 1),
            "Accuracy": acc_list
        })
        df.to_csv(save_path, index=False)

        return acc_list

    def save(self):
        self.model.save(self.model_path)
        print(f"Model saved at {self.model_path}")
        self.is_load = True

    def predict_digit(self, img):
        test_image = img.reshape(-1, 28, 28, 1)
        return self.model.predict(test_image)
class CNNModel(BaseModel):
    label = "cnn"
    def __init__(self, input_shape, model_path=None):
        super().__init__(input_shape, model_path)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))

        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation("softmax"))

        self.model = model
        self.compile()

class CapsuleModel(BaseModel):
    label = "CapsuleModel"

    def __init__(self, input_shape, model_path=None):
        super().__init__(input_shape, model_path)

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        # Conv1, return tensor with shape [batch_size, 20, 20, 256]
        x = layers.Conv2D(256, kernel_size=9, activation='relu')(inputs)
        # primaryCaps pre-processing
        x = layers.Conv2D(256, kernel_size=9, strides=2, activation='relu')(x)

        # PrimaryCaps
        # reshaping, shape： (batch_size, 6*6*32=1152, 8, 1)
        x = layers.Reshape((-1, 8))(x)  # 每个胶囊维度是8
        x = self.squash(x)

        # DigitCaps return shape [batch_size, 10, 16, 1]
        caps_output = CapsLayer(num_capsules=10, dim_capsules=16)(x)

        # Length layer
        output = tf.keras.layers.Lambda(lambda z: tf.norm(z, axis=-1))(caps_output)

        self.model = models.Model(inputs=inputs, outputs=output)

    def squash(self, x, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-9)
        return scale * x