import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import pygame
import io

from MODEL.model import CNNModel
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

    # x_train = tf.keras.utils.normalize(x_train, axis = 1)
    # x_test = tf.keras.utils.normalize(x_test, axis = 1)
    _,x_train = cv2.threshold(x_train,127,255,cv2.THRESH_BINARY)
    _,x_test = cv2.threshold(x_test,127,255,cv2.THRESH_BINARY)
    # reshape the input image to 28x28, channel=1
    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return x_train, x_test

# ===================== image processing ===================== #
def put_label(t_img,label,x,y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_x = int(x) - 10
    l_y = int(y) + 10
    cv2.rectangle(t_img,(l_x,l_y+5),(l_x+35,l_y-35),(0,255,0),-1)
    cv2.putText(t_img,str(label),(l_x,l_y), font,1.5,(255,0,0),1,cv2.LINE_AA)
    return t_img


import numpy as np
import cv2
import math


def image_refiner(gray):
    '''Image preprocessing, resized to 28x28'''
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

    return padded_image


def get_output_image(path):
    img = cv2.imread(path, 0)  # 读取灰度图
    img_org = cv2.imread(path)  # 读取原始彩色图

    ret, thresh = cv2.threshold(img, 127, 255, 0)  # 二值化 像素值大于 127 的会被设为 255（白色），小于 127 的会被设为 0（黑色）。
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测 轮廓 沿着物体的边缘

    for j, cnt in enumerate(contours):
        # 计算轮廓的周长，epsilon 是一个小的阈值，用于调整近似程度
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        # 使用近似多边形来逼近轮廓，epsilon 控制逼近精度
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 获取轮廓的凸包（即围绕轮廓的最小凸形状）
        hull = cv2.convexHull(cnt)
        # 判断该轮廓是否是凸的
        k = cv2.isContourConvex(cnt)
        x, y, w, h = cv2.boundingRect(cnt)  # 获取矩形边界框

        # 如果轮廓是有效的（hierarchy[0][j][3] != -1 表示该轮廓没有父轮廓）并且其尺寸大于阈值，过滤掉小噪声
        if hierarchy[0][j][3] != -1 and w > 10 and h > 10:  # 过滤小的噪声轮廓
            # 在图像上绘制绿色矩形框，标注识别区域
            cv2.rectangle(img_org, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 裁剪数字部分 提取出轮廓内的区域
            roi = img[y:y + h, x:x + w]
            roi = cv2.bitwise_not(roi)  # 反转颜色（黑底白字）符合mnist训练集的要求
            roi = image_refiner(roi)  # 调整大小并填充
            th, fnl = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)  # 再次对裁剪的区域进行二值化，确保其为黑白图像

            # load训练好的model进行数字预测
            model = CNNModel(input_shape=(28, 28, 1))
            pred = model.predict_digit(roi)
            pred_argmax = np.argmax(pred)
            print(pred)

            # 在图像上标注预测结果
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            img_org = put_label(img_org, pred_argmax, x, y)

    return img_org
# ===================== image processing ===================== #

# ===================== windows service ===================== #
# pre defined colors, pen radius and font color
black = [0, 0, 0]
white = [255, 255, 255]
red = [255, 0, 0]
green = [0, 255, 0]
draw_on = False
last_pos = (0, 0)
color = (255, 128, 0)
radius = 7
font_size = 500

#image size
width = 640
height = 640

# initializing screen
screen = pygame.display.set_mode((width*2, height))
screen.fill(white)
pygame.font.init()

def show_output_image(img):
    """Displays the processed image on the right side of the screen."""
    surf = pygame.pixelcopy.make_surface(img)
    surf = pygame.transform.rotate(surf, -270)
    surf = pygame.transform.flip(surf, 0, 1)
    screen.blit(surf, (width+2, 0)) # Display image on the right side

def crope(orginal):
    """Crops the drawn area slightly to remove boundary artifacts."""
    cropped = pygame.Surface((width-5, height-5))
    cropped.blit(orginal, (0, 0), (0, 0, width-5, height-5))
    return cropped

def roundline(srf, color, start, end, radius=1):
    """Draws smooth lines by interpolating between points."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color, (x, y), radius)

def draw_partition_line():
    """Draws a vertical separation line between drawing and output areas."""
    pygame.draw.line(screen, black, [width, 0], [width,height ], 8)
