"""
Выделяет квадрат на картинке содержащий объект
"""
import gc
import os
import cv2
from glob import glob
import xml.etree.ElementTree as ET
from collections import namedtuple
from statistics import mean
import multiprocess as mp
from deeplab3p_segmentation.train import create_dir
import time
from pascal_voc_writer import Writer
counter = 0
cpu_count = 8

# print(mp.cpu_count())
annotation_ext = ".txt"
source_dataset_path = "/mnt/sda2/its/dataset/guardrail_dmg/"
source_path = os.path.join(source_dataset_path, "2")


def load_dataset(_src_path, _ann_ext, _ann_path=None):
    if not _ann_path:
        _ann_path = _src_path
    """ Load the dataset: images and labels """
    _data_x = glob(os.path.join(_src_path, "*.jpg"))
    _data_y = glob(os.path.join(_ann_path, '*'+_ann_ext))
    return _src_path, _ann_path, _data_x, _data_y


result_name = "1_sqr_crop"
result_path = os.path.join(source_dataset_path, result_name)
create_dir(result_path)
Bbox = namedtuple('Bbox', ['xmin','ymin', 'xmax', 'ymax'])


def delete_unused_annotations(src_path, ann_path, img_list, ann_list):
    # deletes symmetric difference files - between images and annotations in source directory
    img_names_set = set([n.split("/")[-1].split(".")[0] for n in img_list])
    ann_names_set = set([n.split("/")[-1].split(".")[0] for n in ann_list])

    diff_names = img_names_set ^ ann_names_set
    print('differences are:', diff_names)
    for diff_name in diff_names:
        # print(diff_name)
        # for filename in glob(os.path.join(src_path, diff_name + '*')):
        #     print('deleting', filename)
        #     # os.remove(filename)
        for filename in glob(os.path.join(ann_path, diff_name + '*')):
            print('deleting', filename)
            os.remove(filename)


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes_list = []
    for boxes in root.iter('object'):
        ann_name = root.find('filename').text
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        bbox = Bbox(xmin, ymin, xmax, ymax)
        bboxes_list.append(bbox)
    return ann_name, bboxes_list

def validate_bbox(bbox_, h_, w_):
    if 0 <= bbox_.xmin <= w_ and \
       0 <= bbox_.ymin <= h_ and \
       0 <= bbox_.xmax <= w_ and \
       0 <= bbox_.ymax <= h_:
        return True
    else:
        return False


def crop_square_image(im_path):
    start = time.time()
    name = im_path.split("/")[-1].split(".")[0]
    print(name)
    # get the bbox
    annotation_path = os.path.join(source_path, name + annotation_ext)
    filename, bboxes = read_content(annotation_path)
    # print(filename, bboxes)
    img = cv2.imread(im_path)
    h, w, c = img.shape
    w_half = w // 2
    # принимаем что ббокс один!
    bbox = bboxes[0]
    gc.collect()

    # проверка что границы ббоксов не выходят за изображение
    is_valid_bbox = validate_bbox(bbox, h, w)
    if not is_valid_bbox:
        print('REMOVE!', annotation_path, im_path)
        print('deleting')
        os.remove(im_path)
        os.remove(annotation_path)
        gc.collect()
        return
    else:

        if (bbox.xmin <= w_half) and (bbox.xmax <= w_half):
            # cv2.imshow('debug', img)
            # cv2.waitKey(0)
            # print(img.shape)
            img = img[:w_half, :w_half]
            print('LEFT. new shape is:', img.shape)
            # cv2.imshow('debug', img)
            # cv2.waitKey(0)
            new_bbox = bbox

        elif (bbox.xmin > w_half) and (bbox.xmax > w_half):
            # cv2.imshow('debug', img)
            # cv2.waitKey(0)
            # print(img.shape)
            img = img[:w_half, w_half:]
            print('RIGHT. new shape is:', img.shape)
            # cv2.imshow('debug', img)
            # cv2.waitKey(0)
            new_bbox = Bbox(bbox.xmin - w_half, bbox.ymin, bbox.xmax - w_half, bbox.ymax)

        else:
            bbox_center_x = round(mean([bbox.xmin, bbox.xmax]))
            x_left_bound = bbox_center_x - w // 4
            x_right_bound = bbox_center_x + w // 4
            # print('sebe!', bbox)
            # img = cv2.rectangle(img, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (40, 150, 60), 3)
            # cv2.imshow('debug', img)
            # cv2.waitKey(0)
            img = img[:w_half, x_left_bound:x_right_bound]
            print('CENTER!. new shape is:', img.shape)
            # cv2.imshow('debug', img)
            # cv2.waitKey(0)
            new_bbox = Bbox(bbox.xmin - x_left_bound, bbox.ymin, bbox.xmax - x_left_bound, bbox.ymax)
            # img = cv2.rectangle(img, (new_bbox.xmin, new_bbox.ymin), (new_bbox.xmax, new_bbox.ymax), (166, 150, 60), 3)
            # cv2.imshow('debug', img)
            # cv2.waitKey(0)

    writer = Writer(os.path.join(result_path, name + ".jpg"), img.shape[1], img.shape[0])
    writer.addObject('guardrail-damage', new_bbox.xmin, new_bbox.ymin, new_bbox.xmax, new_bbox.ymax)
    writer.save(os.path.join(result_path, f'{name}.xml'))
    cv2.imwrite(os.path.join(result_path, name + ".jpg"), img)
    gc.collect()
    return

def check_images_equal_annotations(src_path):
    try:
        src_path, ann_path, data_x_, data_y_ = load_dataset(src_path, ".xml")
        assert (len(data_x_) == len(data_y_))
        print('equal - ok')
        return data_x_, data_y_
    except AssertionError:
        delete_unused_annotations(src_path, ann_path, data_x_, data_y_)
        check_images_equal_annotations(src_path)


def check_images_equal_annotations2(src_path, ann_path):
    try:
        src_path, ann_path, data_x_, data_y_ = load_dataset(src_path, ".txt", ann_path)
        assert (len(data_x_) == len(data_y_))
        print('equal - ok')
        return data_x_, data_y_
    except AssertionError:
        delete_unused_annotations(src_path, ann_path, data_x_, data_y_)
        check_images_equal_annotations(src_path)

def main():
    data_x, data_y = check_images_equal_annotations(source_path)
    p = mp.Pool(6)
    p.map(crop_square_image, data_x)
    p.close()


if __name__ == '__main__':
    main()



