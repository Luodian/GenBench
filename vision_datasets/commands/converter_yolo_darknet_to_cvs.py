"""
LEGACY: Converts YOLO Darknet TXT format to iris format
"""

import argparse
import os
import pathlib
import zipfile

from datetime import datetime
from PIL import Image
from tqdm import tqdm
from typing import List

IMAGE_FILE_EXTENSION = '.jpg'


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Convert YOLO Darknet TXT format to iris format for an object detection dataset.')
    parser.add_argument('-i', '--input', type=pathlib.Path, required=True, help='Path to the directory of image and label data in YOLO format.')
    parser.add_argument('-o', '--output', type=pathlib.Path, help='Path to the output directory to write data to in iris format.')

    zip_parser = parser.add_mutually_exclusive_group(required=False)
    zip_parser.add_argument('--zip', dest='zip', action='store_true', help='Flag to choose to zip the image and label folders (default).')
    zip_parser.add_argument('--no-zip', dest='zip', action='store_false', help='Flag to choose NOT to zip the image and label folders.')
    parser.set_defaults(zip=True)
    return parser


def read_annotation_file(path: pathlib.Path):
    return [list(map(float, bbox.split())) for bbox in path.read_text().splitlines()]


def write_iris_annotation_file(path: pathlib.Path, bboxes: List[List]):
    path.write_text('\n'.join([' '.join(map(str, bbox)) for bbox in bboxes]))


def validate_and_correct_iris_bbox(bbox: List, image_width: int, image_height: int):
    if 0 <= bbox[1] <= image_width and 0 <= bbox[2] <= image_height:
        bbox[3], bbox[4] = min(bbox[3], image_width), min(bbox[4], image_height)
        return list(map(int, bbox))


def convert_yolo_bboxes_to_iris(yolo_bboxes: List[List[float]], image_width: int, image_height: int):
    iris_bboxes = []
    for _class, x_center, y_center, bbox_width, bbox_height in yolo_bboxes:
        iris_bbox = [_class,
                     (x_center - bbox_width / 2) * image_width,
                     (y_center - bbox_height / 2) * image_height,
                     (x_center + bbox_width / 2) * image_width,
                     (y_center + bbox_height / 2) * image_height]
        iris_bboxes += [validate_and_correct_iris_bbox(iris_bbox, image_width, image_height)]
    return list(filter(None, iris_bboxes))


def zip_directory(dirname):
    zip_file = zipfile.ZipFile(f'{dirname}.zip', 'w', zipfile.ZIP_STORED)
    os.chdir(dirname)
    for _, _, files in os.walk(dirname):
        for file in files:
            zip_file.write(file)
    zip_file.close()


def main():
    args = create_arg_parser().parse_args()
    if not args.input.is_dir():
        raise argparse.ArgumentTypeError(f'{args.input} is not a valid directory path')

    if not args.output:
        args.output = args.input.parent / f'{args.input.name}_{datetime.today().strftime("%Y%m%d")}'
    args.output.mkdir(exist_ok=True)  # create output directory if not exists

    im_meta_info_fh = (args.output / 'image_meta_info.txt').open('w')
    datasplits = [subdir.name for subdir in args.input.iterdir() if subdir.is_dir()]

    for datasplit in datasplits:
        input_image_dir, input_label_dir = args.input / datasplit / 'images', args.input / datasplit / 'labels'
        assert len(list(input_image_dir.iterdir())) == len(list(input_label_dir.iterdir()))

        output_image_dir, output_label_dir = args.output / (datasplit + '_images'), args.output / (datasplit + '_labels')
        output_image_dir.mkdir(exist_ok=True)
        output_label_dir.mkdir(exist_ok=True)
        index_fh = (args.output / (datasplit + '_images.txt')).open('w')

        for label_file in tqdm(input_label_dir.iterdir(), desc=datasplit):
            image_fname = label_file.stem + IMAGE_FILE_EXTENSION
            input_image_fp = input_image_dir / image_fname
            image_width, image_height = Image.open(input_image_fp).size
            input_image_fp.replace(output_image_dir / image_fname)  # move image to output image directory

            # write to image meta info and index file
            im_meta_info_fh.write(f'{output_image_dir.name}.zip@{image_fname} {image_width} {image_height}\n')
            index_fh.write(f'{output_image_dir.name}.zip@{image_fname} {output_label_dir.name}.zip@{label_file.name}\n')

            # convert annotation from yolo to iris format
            yolo_bboxes = read_annotation_file(label_file)
            iris_bboxes = convert_yolo_bboxes_to_iris(yolo_bboxes, image_width, image_height)
            write_iris_annotation_file(output_label_dir / label_file.name, iris_bboxes)

        if args.zip:
            zip_directory(output_image_dir)
            zip_directory(output_label_dir)

        index_fh.close()
    im_meta_info_fh.close()


if __name__ == '__main__':
    main()
