import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import pygame
import io
import os

from MODEL.model import CNNModel
# ===================== global configuration ===================== #
IMG_SIZE = 28
# :para black: RGB value for black color used in drawing and UI elements.
black = [0, 0, 0]

# :para white: RGB value for white color, used as background color.
white = [255, 255, 255]

# :para red: RGB value for red color, can be used for alerts or highlights.
red = [255, 0, 0]

# :para green: RGB value for green color, used for bounding boxes and highlights.
green = [0, 255, 0]

# :para draw_on: Flag to indicate if the drawing mode is active.
draw_on = False

# :para last_pos: Stores the last mouse/touch position for continuous drawing.
last_pos = (0, 0)

# :para color: Current drawing color, default is orange (255, 128, 0).
color = (255, 128, 0)

# :para radius: Radius of the drawing brush/stroke.
radius = 7

# :para font_size: Font size for number rendering (if used with pygame.font).
font_size = 500

# ===================== image size configuration ===================== #

# :para width: Width of a single drawing/prediction panel.
width = 640

# :para height: Height of the drawing/prediction panel.
height = 640

# ===================== pygame screen initialization ===================== #

# :para screen: The main pygame display surface initialized to 3*width by height.
screen = pygame.display.set_mode((width * 3, height))

# Fill the screen background with white color initially
screen.fill(white)

# :para pygame.font.init: Initializes the font module in pygame.
pygame.font.init()




# ===================== image processing ===================== #

def imageNormalization(x_train, x_test, mean=0.1037, std=0.3081):
    """
    :para x_train: Input training images (numpy array).
    :para x_test: Input test images (numpy array).
    :return: Normalized and reshaped x_train and x_test images.
    """
    # _, x_train = cv2.threshold(x_train, 127, 255, cv2.THRESH_BINARY)
    # _, x_test = cv2.threshold(x_test, 127, 255, cv2.THRESH_BINARY)

    # # normalization
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Normalize by subtracting the mean and dividing by the standard deviation
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # reshape the input image to 28x28, channel=1
    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return x_train, x_test


def put_label(t_img, label, x, y):
    """
    :para t_img: The image to place the label on (numpy array).
    :para label: The label to be placed on the image.
    :para x: The x-coordinate for the label position.
    :para y: The y-coordinate for the label position.
    :return: The image with the label placed on it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_x = int(x) - 10
    l_y = int(y) + 10
    cv2.rectangle(t_img, (l_x, l_y + 5), (l_x + 35, l_y - 35), (0, 255, 0), -1)
    cv2.putText(t_img, str(label), (l_x, l_y), font, 1.5, (255, 0, 0), 1, cv2.LINE_AA)
    return t_img


def image_refiner(gray):
    """
    :para gray: The grayscale image to be resized and padded (numpy array).
    :return: A padded and resized image of size 28x28.
    """
    org_size = 22  # Original Size
    img_size = 28  # Target size
    rows, cols = gray.shape

    if rows > cols:
        factor = org_size / rows
        rows = org_size
        cols = int(round(cols * factor))
    else:
        factor = org_size / cols
        cols = org_size
        rows = int(round(rows * factor))

    gray = cv2.resize(gray, (cols, rows))

    # Calculate padding
    colsPadding = (int(math.ceil((img_size - cols) / 2.0)), int(math.floor((img_size - cols) / 2.0)))
    rowsPadding = (int(math.ceil((img_size - rows) / 2.0)), int(math.floor((img_size - rows) / 2.0)))

    # Manually apply padding by creating a new array and inserting the image
    padded_image = np.zeros((img_size, img_size), dtype=gray.dtype)

    # Insert the resized image into the center of the padded image
    padded_image[rowsPadding[0]:rowsPadding[0] + rows, colsPadding[0]:colsPadding[0] + cols] = gray

    # Ensure the image has the correct shape (28, 28, 1)
    padded_image = np.expand_dims(padded_image, axis=-1)

    return padded_image


def printPred_array(pred_array):
    """
    :para pred_array: A numpy array of predictions (each containing 10 probability values for each sample).
    :return: Prints the predicted digit and probabilities for each sample.
    """
    for i in range(pred_array.shape[0]):  # Iterate through each sample
        current_pred = pred_array[i, 0, :]
        pred_argmax = np.argmax(current_pred)

        print(f"Sample {i + 1}:")
        print(f"Predicted digit (argmax): {pred_argmax}")
        print(f"Probabilities: {current_pred}")


def get_output_image(path, model):
    """
    :para path: The file path to the image to be processed.
    :para model: The trained model for predicting the digit.
    :return: The processed image with bounding boxes and predicted labels, along with the prediction array.
    """
    img = cv2.imread(path, 0)  # Read the grayscale image
    img_org = cv2.imread(path)  # Read the original colored image

    ret, thresh = cv2.threshold(img, 127, 255, 0)  # Threshold the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    pred_list = []
    for j, cnt in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull = cv2.convexHull(cnt)
        k = cv2.isContourConvex(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if hierarchy[0][j][3] != -1 and w > 10 and h > 10:  # Filter out small noise contours
            cv2.rectangle(img_org, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi = img[y:y + h, x:x + w]
            roi = cv2.bitwise_not(roi)
            roi = image_refiner(roi)  # Refine the image

            # Ensure the image is in the correct format (28, 28, 1)
            roi = np.expand_dims(roi, axis=0)

            pred = model.predict_digit(roi)
            pred_list.append(pred)
            pred_argmax = np.argmax(pred)

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            img_org = put_label(img_org, pred_argmax, x, y)

    pred_array = np.array(pred_list)
    printPred_array(pred_array)
    return img_org, pred_array


# ===================== image processing ===================== #

# ===================== windows service ===================== #

def show_prediction_chart(pred):
    """
    :para pred: Prediction array with probabilities for each digit.
    :return: Displays a bar chart showing digit probabilities.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    x = np.arange(10)  # 0-9 digits
    ax.bar(x, pred.flatten(), color='blue')
    ax.set_xticks(x)
    ax.set_xlabel("Digits")
    ax.set_ylabel("Probability")
    ax.set_title("Digit Probabilities")

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    img = pygame.image.load(buf)
    buf.close()

    surf = pygame.transform.scale(img, (width, height))
    screen.blit(surf, (width * 2 + 2, 0))


def show_output_image(img):
    """
    :para img: The processed image to be displayed.
    :return: Displays the processed image on the screen.
    """
    surf = pygame.pixelcopy.make_surface(img)
    surf = pygame.transform.rotate(surf, -270)
    surf = pygame.transform.flip(surf, 0, 1)
    screen.blit(surf, (width + 2, 0))


def show_combined_output(img, pred):
    """
    :para img: The processed image to be displayed.
    :para pred: The prediction array for the displayed image.
    :return: Displays the image and the prediction chart side by side.
    """
    img_surf = pygame.pixelcopy.make_surface(img)
    img_surf = pygame.transform.rotate(img_surf, -270)
    img_surf = pygame.transform.flip(img_surf, 0, 1)

    fig, ax = plt.subplots(figsize=(3, 3))
    x = np.arange(10)

    if pred.size > 0:
        ax.bar(x, pred[0, 0], color='blue')
    else:
        print("Prediction array is empty, skipping chart.")

    ax.set_xticks(x)
    ax.set_xlabel("Digits")
    ax.set_ylabel("Probability")
    ax.set_title("Digit Probabilities")

    os.makedirs("ASSETS", exist_ok=True)
    plt.savefig("ASSETS/Fig.png", format="PNG", bbox_inches="tight")
    plt.close(fig)

    try:
        chart_surf = pygame.image.load("ASSETS/Fig.png")
        chart_surf = pygame.transform.scale(chart_surf, (width, height))
        screen.blit(chart_surf, (width * 2 + 2, 0))
    except pygame.error as e:
        print(f"Error: Failed to load chart image from ASSETS/Fig.png: {e}")

    screen.blit(img_surf, (width + 2, 0))

    pygame.display.update([(width + 2, 0, width, height), (width * 2 + 2, 0, width, height)])


def crope(original):
    """
    :para original: The original pygame surface to be cropped.
    :return: A cropped version of the original surface.
    """
    cropped = pygame.Surface((width - 5, height - 5))
    cropped.blit(original, (0, 0), (0, 0, width - 5, height - 5))
    return cropped


def roundline(srf, color, start, end, radius=1):
    """
    :para srf: The surface on which to draw the line.
    :para color: The color of the line.
    :para start: The start point of the line (tuple).
    :para end: The end point of the line (tuple).
    :para radius: The radius of the line, defaults to 1.
    :return: None. Draws the line on the surface.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color, (x, y), radius)


def draw_partition_line():
    """
    :return: Draws vertical partition lines to divide the sections of the window.
    """
    pygame.draw.line(screen, black, [width, 0], [width, height], 8)
    pygame.draw.line(screen, black, [width * 2, 0], [width * 2, height], 8)
