from split_image import delete_unused_annotations, load_dataset, check_images_equal_annotations
import os
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=False, help="path to dataset dir")
args = vars(ap.parse_args())
source_path = args["data_dir"]
source_path = '/mnt/sda2/its/dataset/guardrail_dmg/task_sqr_crop'

if __name__ == '__main__':
    check_images_equal_annotations(source_path)