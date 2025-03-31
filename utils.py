import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
IMG_SIZE = 28
def checkMATPLOTLIB(image, label):
    plt.figure(figsize=(12, 4))  # 设置整体图像大小

    # 左侧：默认颜色（伪彩色）
    plt.subplot(1, 2, 1)  # 1行2列，第1个
    plt.imshow(image)
    plt.title(f"Original (Label: {label})")
    plt.axis("off")

    # 右侧：灰度显示
    plt.subplot(1, 2, 2)  # 1行2列，第2个
    plt.imshow(image, cmap= 'gray')
    plt.title(f"Grayscale (Label: {label})")
    plt.colorbar()  # 显示颜色条
    plt.axis("off")

    plt.show()


def checkMNISTdata(x_train, y_train, x_test, y_test):
    print(f"x_train shape: {x_train.shape}") # samples
    print(f"y_train shape: {y_train.shape}") # labels
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_test.shape}")

def imageNormalization(x_train, x_test):
    # reshape the input image to 28x28, channel=1
    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_train = tf.keras.utils.normalize(x_train, axis = 1)
    x_test = tf.keras.utils.normalize(x_test, axis = 1)
    return x_train, x_test