import os
import random

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

import shepards_distotrion
from deeplab3p_segmentation.metrics import dice_coef, iou, dice_loss
from deeplab3p_segmentation.train import create_dir
import math
from patchify import patchify
from sklearn.linear_model import LinearRegression
from line_approximator import Line
from shepards_distotrion import get_shepards_distortion, PointTransition, BoundingBox
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pascal_labeling import create_pascal_label
""" Global parameters """
height_net_input = 512
width_net_input = 512


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

def show_debug_image_and_line(im_, li_):
    im_ = cv2.line(im_, (li_.x_min, li_.y_min), (li_.x_max, li_.y_max), (0.9, 0.1, 0.1), 2)
    cv2.imshow("debug", im_)
    cv2.waitKey(0)

def convert_to_original_coords(distortionn_points, w_scale, h_scale):
    """ конвертирует координаты из уменьшенного изображения в первоначальное"""
    new_distortion_points = [PointTransition(
        point.x * w_scale,
        point.y * h_scale,
        point.i * w_scale,
        point.j * h_scale
    ) for point in distortionn_points]
    return new_distortion_points

def get_longest_line(image, model_):
    """принимает картинку по пути path, модель и возвращает объект
     самой длинной найденной на фото аппроксимирующей линии барьера"""
    area_thres = 5
    resized_image = cv2.resize(image, (width_net_input, height_net_input))
    resized_image = resized_image / 255.0
    resized_image = resized_image.astype(np.float32)
    resized_image = np.expand_dims(resized_image, axis=0)

    """ Prediction """
    prediction = model_.predict(resized_image)[0]
    prediction_BGR = cv2.cvtColor(255 * prediction, cv2.COLOR_GRAY2BGR)
    prediction_GRAY = cv2.cvtColor(prediction_BGR, cv2.COLOR_BGR2GRAY)
    prediction_blur = cv2.GaussianBlur(prediction_GRAY, (7, 7), 1)
    prediction_canny = cv2.Canny(prediction_blur.astype(np.uint8), 50, 350)
    kernel = np.ones((3, 3), np.uint8)
    # variant
    # imgDialation = cv2.dilate(prediction_canny, kernel, iterations=1)
    contours_, hierarchy_ = cv2.findContours(prediction_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    approx_lines_ = []
    for cnt_ in contours_:
        area_ = cv2.contourArea(cnt_)
        if area_ > area_thres:
            peri_ = cv2.arcLength(cnt_, True)
            approx_ = cv2.approxPolyDP(cnt_, 0.0001 * peri_, True)
            x_dots = approx_[:, 0, 0]
            y_dots = approx_[:, 0, 1]
            reg = LinearRegression().fit(x_dots.reshape(-1, 1), y_dots)
            print("debug. lineapprox score is:", reg.score(x_dots.reshape(-1, 1), y_dots))
            x_min = int(min(x_dots))
            x_max = int(max(x_dots))
            y_min = round(reg.predict((np.array(x_min)).reshape(-1, 1))[0])
            y_max = round(reg.predict((np.array(x_max)).reshape(-1, 1))[0])
            approx_lines_.append(Line(x_min, y_min, x_max, y_max, reg.coef_, reg.intercept_))
    if len(approx_lines_) == 0:
        return None, None
    longest_line = max(approx_lines_)

    # show_debug_image_and_line(resized_image[0], longest_line)

    return longest_line, prediction


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
# data_x = glob("/mnt/sda2/shit_dataset/dataset/JPEGImages/*")


def main():
    for path in tqdm(data_x, total=len(data_x)):
        print("DEBUG:", path)
        """ Extracting name """
        name = path.split("/")[-1].split(".")[0]
        filename = path.split("/")[-1]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        h_scale = h / height_net_input
        w_scale = w / width_net_input
        resized_image = cv2.resize(image, (width_net_input, height_net_input))
        resized_image = resized_image / 255.0
        resized_image = resized_image.astype(np.float32)
        resized_image = np.expand_dims(resized_image, axis=0)

        longest_line, prediction = get_longest_line(image, model)
        if longest_line is None or prediction is None:
            continue
        mask_original_size = cv2.resize(prediction * 255, (w, h))
        mask_path = path + "mask.jpg"
        cv2.imwrite(mask_path, mask_original_size)




        im_ = cv2.line(resized_image[0], (longest_line.x_min, longest_line.y_min), (longest_line.x_max, longest_line.y_max), (0.9, 0.1, 0.1), 2)
        # cv2.imshow("debug", im_)
        # cv2.waitKey(0)

        # вычислить координаты точек переноса для случайного числа
        distortionn_points = get_shepards_distortion(longest_line, random.choice(range(1, 4)))
        distortionn_points_original_scale = convert_to_original_coords(distortionn_points, w_scale, h_scale)

        bbox, saved_image_name = shepards_distotrion.shepards_distortion_multipoint(path, 0.7, distortionn_points_original_scale, save=True, draw=True)
        bbox, saved_mask_name = shepards_distotrion.shepards_distortion_multipoint(mask_path, 0.7, distortionn_points_original_scale, save=True, draw=False)

        bounded_dist_mask = cv2.imread(saved_mask_name, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("debug", bounded_dist_mask)
        # cv2.waitKey(0)

        bounded_dist_mask = bounded_dist_mask[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]
        # cv2.imshow("debug", bounded_dist_mask)
        # cv2.waitKey(0)
        bounded_dist_mask = cv2.GaussianBlur(bounded_dist_mask, (7, 7), 1)
        bounded_dist_mask = cv2.Canny(bounded_dist_mask.astype(np.uint8), 50, 350)
        # cv2.imshow("debug", bounded_dist_mask)
        # cv2.waitKey(0)
        kernel = np.ones((5, 5), np.uint8)
        # variant
        bounded_dist_mask = cv2.dilate(bounded_dist_mask, kernel, iterations=2)
        # cv2.imshow("debug", bounded_dist_mask)
        # cv2.waitKey(0)
        contours_, hierarchy_ = cv2.findContours(bounded_dist_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours_, key=lambda c: cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(bounded_dist_mask, (x, y), (x + w, y + h), (255), 2)
        # cv2.imshow("debug", bounded_dist_mask)
        # cv2.waitKey(0)
        image_2 = cv2.imread(saved_image_name, cv2.IMREAD_COLOR)
        cv2.rectangle(image_2, (bbox.x_min + x, bbox.y_min + y), (bbox.x_min + x + w, bbox.y_min + y + h), (255, 160, ), 4)
        cv2.imwrite(saved_image_name, image_2)

        # debug
        for dist_point in distortionn_points:
            im_ = cv2.line(im_, (dist_point.x, dist_point.y),
                           (dist_point.i, dist_point.j), (0.1, 0.7, 0.9), 1)
        # cv2.imshow("debug", im_)
        # cv2.waitKey(0)




main()

    # delta_x = 4 * math.log(longest_line.distance(), 1.7) / math.log(abs(-1 / longest_line.coef), math.e)
    # print("Distance logarithm:", delta_x)
    # x_mid0 = int(longest_line.mid_point[0])
    # y_mid0 = int(longest_line.mid_point[1])
    # x_mid1 = int(x_mid0 + (delta_x * math.copysign(1, longest_line.coef)))
    # y_mid1 = int(longest_line.calculate_normal(x_mid1))
    # print()
    # x[0] = cv2.line(x[0], (x_mid0, y_mid0), (x_mid1, y_mid1),
    #                 (0.05, 0.6, 0.9), 2)
    #
    # bbox = shepards_distotrion.shepards_distortion(path, x_mid0 * width_scale, y_mid0 * height_scale,
    #                                         x_mid1 * width_scale, y_mid1 * height_scale, "0.6")
    # # create_pascal_label(filename, w, h, bbox)
    #
    # # cv2.imshow("result", x[0])
    # # cv2.waitKey(0)
    #
    # result = x[0] * 255
    # result = cv2.resize(result, [w, h])
    # cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (160, 50, 28), 3)
    #
    # cv2.imwrite(os.path.join("debug", name+"_debug_.jpg"), result)
    # y = cv2.resize(y, (w, h))
    # y = np.expand_dims(y, axis=-1)




