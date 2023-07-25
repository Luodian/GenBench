"""
Converts tsv format to coco format
ic: img_id  [{"class": class_name}, ....]    img_data_base64
od: img_id  [{"class": class_name, "rect": [L, T, R, B], "diff": 0}, ...] img_data_base64
caption: img_id [{"caption": caption}, ....]    img_data_base64
"""

import json
import os
import pathlib
import re
from tqdm import tqdm

from vision_datasets.common.util import write_to_json_file_utf8


from .utils import verify_and_correct_box_or_none, guess_encoding, Base64Utils, zip_folder, set_up_cmd_logger, TSV_FORMAT_LTRB, TSV_FORMAT_LTWH_NORM

logger = set_up_cmd_logger(__name__)

pattern = re.compile(r'[\./\\\@]')


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert tsv data to coco format.')
    parser.add_argument('-t', '--tsvs', required=True, nargs='+', help='Tsv files to convert.')
    parser.add_argument('-c', '--task', required=True, type=str, help='type of tasks.', choices=['ic', 'od', 'caption'])
    parser.add_argument('-l', '--labelmap', type=pathlib.Path, default=None)
    parser.add_argument('-f', '--format', type=str, default=TSV_FORMAT_LTRB, choices=[TSV_FORMAT_LTRB, TSV_FORMAT_LTWH_NORM])
    parser.add_argument('-d', '--difficulty', type=bool, default=False, help='Include difficulty boxes or not.')
    parser.add_argument('-z', '--zip', type=bool, default=False, help='Zip the image and label folders or not.')
    parser.add_argument('-o', '--output_folder', type=pathlib.Path, required=True, help='Output folder for images.')

    return parser


def get_label_id(label_name, label_name_to_idx, labelmap_exists):
    if label_name not in label_name_to_idx:
        if labelmap_exists:
            raise ValueError(f'Illegal class {label_name}, not in provided labelmap.')

        label_name_to_idx[label_name] = len(label_name_to_idx)

    return label_name_to_idx[label_name]


def main():
    args = create_arg_parser().parse_args()
    logger.info(args.__dict__)
    image_folder_name = args.output_folder

    if not image_folder_name.exists():
        os.mkdir(image_folder_name)

    labelmap_path = args.labelmap
    labelmap_path_exists = labelmap_path and labelmap_path.exists()
    if args.task != 'caption':
        if labelmap_path_exists:
            logger.info(f'Labelmap {labelmap_path} exists.')
            categories = [x.strip() for x in labelmap_path.read_text(encoding='utf-8').split('\n') if x.strip()]

            if not categories:
                raise ValueError(f'Empty labelmap {args.labelmap}')

            label_name_to_idx = {x: i for i, x in enumerate(categories)}
            categories = [{'name': x, 'id': i + 1} for i, x in enumerate(categories)]
        else:
            if args.task == 'ic':
                logger.warn('labelmap recommended to exist for ic.')

            label_name_to_idx = dict()
            categories = None

    for tsv_file_name in args.tsvs:
        images = []
        annotations = []
        with open(tsv_file_name, 'r', encoding=guess_encoding(tsv_file_name)) as file_in:
            for img_idx, img_info in tqdm(enumerate(file_in), desc=f'Processing {tsv_file_name}.'):
                img_id, labels, img_b64 = img_info.split('\t')

                with Base64Utils.b64_str_to_pil(img_b64) as img:
                    w, h = img.size
                    img_format = img.format

                # image data => image file
                img_file_name = pattern.sub('_', img_id) + '.' + img_format
                img_file_path = image_folder_name / img_file_name
                Base64Utils.b64_str_to_file(img_b64, img_file_path)

                img_info_dict = {'id': img_idx+1, 'width': w, 'height': h, 'file_name': str(img_file_path.as_posix()), 'zip_file': f'{image_folder_name}.zip'}
                images.append(img_info_dict)

                # image info => index file
                lp = f'File: {tsv_file_name}, Line {img_idx+1}: '

                labels = json.loads(labels)
                for label in labels:
                    if args.task == 'caption':
                        annotations.append({'id': len(annotations) + 1, 'caption': label['caption'], 'image_id': img_idx+1})
                        continue

                    label_idx = get_label_id(label['class'], label_name_to_idx, labelmap_path_exists)
                    annotation = {'id': len(annotations) + 1, 'category_id': label_idx + 1, 'image_id': img_idx+1}
                    if args.task == 'od':
                        # labels => files
                        difficulty = label['diff'] if 'diff' in label else 0
                        if difficulty > 0 and not args.difficulty:
                            continue

                        box = verify_and_correct_box_or_none(lp, label['rect'], args.format, w, h)
                        if box is None:
                            logger.warn(f'Ignore bbox {box}, as it does not pass check.')
                            continue

                        annotation['bbox'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]

                    annotations.append(annotation)

                if (img_idx+1) % 2000 == 0:
                    logger.info(f'Processed {img_idx} images.')

        coco_file_name = pathlib.Path(os.path.splitext(os.path.basename(tsv_file_name))[0] + '.json')
        if args.task != 'caption':
            categories = categories or [{'id': idx + 1, 'name': name} for name, idx in label_name_to_idx.items()]
            write_to_json_file_utf8({'images': images, 'annotations': annotations, 'categories': categories}, coco_file_name)
        else:
            write_to_json_file_utf8({'images': images, 'annotations': annotations}, coco_file_name)

    if labelmap_path and not labelmap_path_exists and args.task != 'caption':
        with open(labelmap_path, 'w') as labelmap_out:
            idx_to_labels = {label_name_to_idx[key]: key for key in label_name_to_idx}
            for i in range(len(idx_to_labels)):
                labelmap_out.write(idx_to_labels[i] + '\n')

    if args.zip:
        logger.info(f'Zip folder "{image_folder_name}".')
        zip_folder(image_folder_name)


if __name__ == '__main__':
    main()
