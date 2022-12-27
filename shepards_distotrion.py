from collections import namedtuple
from wand.color import Color
from wand.image import Image
import line_approximator
import cv2
import random
import math
from math import hypot, log
from itertools import chain


PointTransition = namedtuple('Point', ['x', 'y', 'i', 'j'])
BoundingBox = namedtuple('BoundingBox', ['x_min', 'y_min', 'x_max', 'y_max'])

def shepards_distortion(filename, x0, y0, x1, x2, power):
    with Image(filename=filename) as img:
        width = img.width
        height = img.height
        img.artifacts['shepards:power'] = power
        alpha = PointTransition(0, 0, 0, 0)
        beta = PointTransition(width, 0, width, 0)
        gamma = PointTransition(0, height, 0, height)
        delta = PointTransition(width, height, width, height)
        f = PointTransition(x0, y0, x1, x2)
        args = (
            *f,
            *alpha,
            *beta,
            *gamma,
            *delta,
        )
        img.distort('shepards', args)
        img.format = 'jpeg'
        saved_image_name = f'{filename}_processed.jpg'
        img.save(filename=saved_image_name)

        bbox_scale_factor = 2.2
        bbox_center_x = (f.x + f.i) / 2
        bbox_center_y = (f.y + f.j) / 2
        distortion_distance = hypot((f.x - f.i), (f.y - f.j))
        distortion_distance_x = abs(f.x - f.i)
        distortion_distance_y = abs(f.y - f.j)
        distortion_power = float(power)
        bbox_x_min = int(f.i - distortion_distance_x * bbox_scale_factor * distortion_power * 0.6)
        bbox_y_min = int(bbox_center_y - distortion_distance_y * bbox_scale_factor * distortion_power * 2)
        bbox_x_max = int(f.i + distortion_distance_x * bbox_scale_factor * distortion_power * 0.6)
        bbox_y_max = int(bbox_center_y + distortion_distance_y * bbox_scale_factor * distortion_power * 2)

    image = cv2.imread(saved_image_name, cv2.IMREAD_COLOR)
    cv2.rectangle(image, (bbox_x_min, bbox_y_min), (bbox_x_max, bbox_y_max), (70, 160, 200), 2)
    cv2.imwrite(saved_image_name, image)
    return bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max


def shepards_distortion_multipoint(filename, power, distortion_points, save_bbox=True, draw=True):
    power = str(power)
    with Image(filename=filename) as img:
        width = img.width
        height = img.height
        img.artifacts['shepards:power'] = power
        alpha = PointTransition(0, 0, 0, 0)
        beta = PointTransition(width, 0, width, 0)
        gamma = PointTransition(0, height, 0, height)
        delta = PointTransition(width, height, width, height)

        args = (
            *chain(*distortion_points),
            *alpha,
            *beta,
            *gamma,
            *delta,
        )
        img.distort('shepards', args)
        saved_image_name = f'{filename}_processed{str(power)}.jpg'
        img.format = 'jpeg'
        img.save(filename=saved_image_name)
        image = cv2.imread(saved_image_name, cv2.IMREAD_COLOR)

        bboxes = []
        for f in distortion_points:
            bbox_scale_factor = 2.2
            bbox_center_x = (f.x + f.i) / 2
            bbox_center_y = (f.y + f.j) / 2
            distortion_distance = hypot((f.x - f.i), (f.y - f.j))
            distortion_distance_x = abs(f.x - f.i)
            distortion_distance_y = abs(f.y - f.j)
            distortion_power = float(power)
            bbox_x_min = int(f.i - distortion_distance * float(power))
            bbox_y_min = int(bbox_center_y - distortion_distance_y * bbox_scale_factor * distortion_power * 1.3)
            bbox_x_max = int(f.i + distortion_distance * float(power))
            bbox_y_max = int(f.y + distortion_distance_y * bbox_scale_factor * distortion_power * 2.5)
            bboxes.append(BoundingBox(bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max))
            if draw:
                cv2.line(image, (int(f.x), int(f.y)), (int(f.i), int(f.j)), (0, 180, 230), 1)
                cv2.rectangle(image, (bbox_x_min, bbox_y_min), (bbox_x_max, bbox_y_max), (0, 0, 255), 3)

        #calculating result bbox
        bbox_x_min = max(0, min(bboxes, key=lambda bbox: min(bbox.x_min, bbox.x_max)).x_min)
        bbox_y_min = min(bboxes, key=lambda bbox: min(bbox.y_min, bbox.y_max)).y_min
        bbox_x_max = max(bboxes, key=lambda bbox: max(bbox.x_min, bbox.x_max)).x_max
        bbox_y_max = max(bboxes, key=lambda bbox: max(bbox.y_min, bbox.y_max)).y_max
        cv2.rectangle(image, (bbox_x_min, bbox_y_min), (bbox_x_max, bbox_y_max), (255, 0, 60), 1)
    if save_bbox:
        cv2.imwrite(saved_image_name, image)
    # return bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max
    return BoundingBox(bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max), saved_image_name


def get_shepards_distortion(longest_line, deform_count: int):
    """точки в относительных координатах где применяем искажение"""
    positions = []
    line_sign = math.copysign(1, longest_line.coef)
    step = 0.06
    for i in range(deform_count):
        """ 0.50, 0.55, 0.60, 0.65"""
        positions.append(0.5 - (4 * line_sign * step) - (line_sign * step * i))
    transition_points = []
    """величина искажения"""
    delta_x = 7 * math.log(longest_line.distance(), 1.7)
    # delta_x = 4 * math.log(longest_line.distance(), 1.7) / math.log(abs(-1 / longest_line.coef), math.e)
    for pos in positions:
        transition_points.append(longest_line.calculate_tp_at_pos(pos, delta_x * line_sign))

    return transition_points
