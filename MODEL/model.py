
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

def cnnModel(input_shape):
    # Initialize a Sequential model
    model = Sequential()

    # First Convolutional layer
    model.add(Conv2D(64, (3, 3), input_shape=input_shape))  # Apply 64 filters of size (3, 3)
    model.add(Activation("relu"))  # Use ReLU activation
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Apply max pooling with a (2, 2) pool size

    # Second Convolutional layer
    model.add(Conv2D(64, (3, 3)))  # Apply 64 filters of size (3, 3)
    model.add(Activation("relu"))  # Use ReLU activation
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Apply max pooling with a (2, 2) pool size

    # Third Convolutional layer
    model.add(Conv2D(64, (3, 3)))  # Apply 64 filters of size (3, 3)
    model.add(Activation("relu"))  # Use ReLU activation
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Apply max pooling with a (2, 2) pool size

    # Fully Connected (FC) layer 1 (flatten the 3D output from previous layers)
    model.add(Flatten())  # Flatten the 3D output into 1D
    model.add(Dense(64))  # Dense layer with 64 neurons
    model.add(Activation("relu"))  # Use ReLU activation

    # Fully Connected (FC) layer 2
    model.add(Dense(32))  # Dense layer with 32 neurons
    model.add(Activation("relu"))  # Use ReLU activation

    # Fully Connected (FC) layer 3 (output layer with 10 neurons for classification)
    model.add(Dense(10))  # Dense layer with 10 neurons (one for each class)
    model.add(Activation("softmax"))  # Use Softmax activation for multi-class classification

    return model  # Return the constructed model
