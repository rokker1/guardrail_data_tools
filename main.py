import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from deeplab3p_segmentation.metrics import dice_coef, iou, dice_loss
from deeplab3p_segmentation.train import create_dir
import math
from patchify import patchify
from sklearn.linear_model import LinearRegression
import line_approximator
import shepards_distotrion
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

""" Global parameters """
H = 512
W = 512


def split_image(image_, patch_height, patch_width):
    """define shape - ширина изображения масштабируется до 2 * (размер патча)
        высота изображения изменятеся пропорционально, но не меньше размера
        высоты патча"""
    old_image_height, old_image_width, channel_count = image.shape
    target_width = 2 * patch_width
    resize_coefficient = old_image_width / target_width
    new_height = round(old_image_height / resize_coefficient)
    resized_image = cv2.resize(image, (target_width, max(new_height, patch_height)))

    """
    define borders
    low_border_coef - отступ от верхнего края изображения для обрезки
    снизу картинки отрезается 5%.
    высота патча фиксирована. сверху картинки отрезается то что осталось от 5% нижней границы + высота патча 
    """
    low_border_coef = 0.95
    if new_height > patch_height:
        low_border = round(low_border_coef * new_height)
    if low_border < patch_height:
        low_border = patch_height
    high_border = max(0, low_border - patch_height)
    resized_image = resized_image[high_border:low_border, :, :]
    patch_shape = (patch_height, patch_width, channel_count)
    step = patch_width
    image_patches = patchify(resized_image, patch_shape, step=step)
    return image_patches



""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Directory for storing files """
create_dir("test_images/mask")

""" Папка с проектом"""
project_path = "/mnt/sda2/source/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow/"

""" Loading model """
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model(project_path + "files/model.h5")

""" Load the dataset """
data_x = glob("test_images/image/*")

for path in tqdm(data_x, total=len(data_x)):
    """ Extracting name """
    name = path.split("/")[-1].split(".")[0]
    filename = path.split("/")[-1]
    """ Reading the image """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    # patches = split_image(image, H, W)

    width_scale = w / W  # > 1
    height_scale = h / H  # > 1

    x = cv2.resize(image, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    """ Prediction """
    y = model.predict(x)[0]
    mask_BGR = cv2.cvtColor(255 * y, cv2.COLOR_GRAY2BGR)
    mask_GRAY = cv2.cvtColor(mask_BGR, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("result", mask_GRAY)
    # cv2.waitKey(0)
    mask_blur = cv2.GaussianBlur(mask_GRAY, (7, 7), 1)
    mask_canny = cv2.Canny(mask_blur.astype(np.uint8), 50, 350)
    kernel = np.ones((3, 3), np.uint8)
    imgDialation = cv2.dilate(mask_canny, kernel, iterations=1)
    imgEroded = cv2.erode(imgDialation, kernel, iterations=1)
    # cv2.imshow("result", mask_canny)
    # cv2.imshow("result1", imgDialation)
    # cv2.imshow("result2", imgEroded)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(mask_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    approx_lines = []
    for cnt in contours:
        cv2.drawContours(x[0], cnt, -1, (0, 255, 0), 1)
        area = cv2.contourArea(cnt)
        if area > 0:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.0001 * peri, True)
            cv2.drawContours(x[0], approx, -1, (0, 0, 255), 1)
            x_dots = approx[:, 0, 0]
            y_dots = approx[:, 0, 1]
            reg = LinearRegression().fit(x_dots.reshape(-1, 1), y_dots)

            print("debug. lineapprox score is:", reg.score(x_dots.reshape(-1, 1), y_dots))
            x_min = int(min(x_dots))
            x_max = int(max(x_dots))
            y_min = round(reg.predict((np.array(x_min)).reshape(-1 , 1))[0])
            y_max = round(reg.predict((np.array(x_max)).reshape(-1 , 1))[0])
            approx_lines.append(line_approximator.Line(x_min, y_min, x_max, y_max, reg.coef_, reg.intercept_))


    print("Image is:", name, 'count of approx lines is:', len(approx_lines), '\n'
          "and the length of the longest line is:", max(approx_lines).distance())
    longest_line = max(approx_lines)
    x[0] = cv2.line(x[0], (longest_line.x_min, longest_line.y_min), (longest_line.x_max, longest_line.y_max), (0.9, 0.2, 0.2), 1)


    some_number = 6 * math.log(longest_line.distance(), 1.7) / math.log(abs(-1 / longest_line.coef), math.e)
    print("Distance logarithm:", some_number)
    x_mid0 = int(longest_line.mid_point[0])
    y_mid0 = int(longest_line.mid_point[1])
    x_mid1 = int(x_mid0 + (some_number * math.copysign(1, longest_line.coef)))
    y_mid1 = int(longest_line.calculate_normal(x_mid1))
    print()
    x[0] = cv2.line(x[0], (x_mid0, y_mid0), (x_mid1, y_mid1),
                    (0.05, 0.6, 0.9), 2)

    shepards_distotrion.shepards_distortion(path, x_mid0 * width_scale, y_mid0 * height_scale,
                                            x_mid1 * width_scale, y_mid1 * height_scale, "1")


    cv2.imshow("result", x[0])
    cv2.waitKey(0)

    result = x[0] * 255
    result = cv2.resize(result, [w, h])
    cv2.imwrite(name+"_debug_.jpg", result)
    # y = cv2.resize(y, (w, h))
    # y = np.expand_dims(y, axis=-1)






