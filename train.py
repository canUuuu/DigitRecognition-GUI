from MODEL.model import CNNModel
from utils import *
import tensorflow as tf

# ===================== Preparation =====================
# Load the MNIST dataset from local keras storage
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape the images to [num_samples, 28, 28, 1]
x_train, x_test = imageNormalization(x_train, x_test)
# ===================== Preparation =====================

def main():
    # ===================== Train / Load Model =====================
    model_path = "MODEL/cnn_model.h5"

    # Initialize the CNN model with input shape and model path
    model = CNNModel(x_train.shape[1:], model_path=model_path)
    model.summary()

    # If model is not loaded, compile and train it
    if not model.is_load:
        model.compile(
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.train(x_train, y_train, epochs=20, x_test=x_test, y_test=y_test)
        # model.save()  # Save the trained model

    # Evaluate the model on the test set
    # model.evaluate(x_test, y_test)
    # ===================== Train / Load Model =====================

# Entry point of the script
if __name__ == "__main__":
    main()
