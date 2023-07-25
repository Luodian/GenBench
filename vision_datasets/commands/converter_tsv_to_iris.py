"""
LEGACY: Converts tsv format to iris format
"""

import json
import os

from .utils import guess_encoding, Base64Utils, zip_folder, TSV_FORMAT_LTRB, TSV_FORMAT_LTWH_NORM, verify_and_correct_box_or_none, set_up_cmd_logger

logger = set_up_cmd_logger(__name__)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert detection dataset in tsv to iris format.')
    parser.add_argument('-t', '--tsvs', required=True, nargs='+', help='Tsv files to convert.')
    parser.add_argument('-l', '--labelmap', type=str, default='labelmap.txt')
    parser.add_argument('-f', '--format', type=str, default=TSV_FORMAT_LTRB, choices=[TSV_FORMAT_LTRB, TSV_FORMAT_LTWH_NORM])
    parser.add_argument('-d', '--difficulty', type=bool, default=False, help='Include difficulty boxes or not.')
    parser.add_argument('-z', '--zip', type=bool, default=True, help='Zip the image and label folders or not.')

    return parser


def main():
    args = create_arg_parser().parse_args()
    logger.info(args.__dict__)
    image_folder_name = 'images'
    if not os.path.exists(image_folder_name):
        os.mkdir(image_folder_name)
    label_folder_name = 'labels'
    if not os.path.exists(label_folder_name):
        os.mkdir(label_folder_name)

    labelmap_exists = os.path.exists(args.labelmap)
    if labelmap_exists:
        logger.info(f'Labelmap {args.labelmap} exists.')
        with open(args.labelmap, 'r') as labelmap_in:
            label_name_to_idx = {x.strip(): i for i, x in enumerate(labelmap_in.readlines())}
            if len(label_name_to_idx) == 0:
                raise Exception(f'Empty labelmap {args.labelmap}')
    else:
        logger.info(f'Labelmap {args.labelmap} does not exist, created on the fly.')
        label_name_to_idx = dict()

    img_meta_data_file_name = 'image_meta_data.txt'
    with open(img_meta_data_file_name, 'w') as image_meta_data_out:
        for tsv_file_name in args.tsvs:
            index_file_name = os.path.splitext(os.path.basename(tsv_file_name))[0] + '.txt'
            line_idx = 1
            with open(index_file_name, 'w') as index_file_out:
                with open(tsv_file_name, 'r', encoding=guess_encoding(tsv_file_name)) as file_in:
                    logger.info(f'Processing {tsv_file_name}.')
                    for img_info in file_in:
                        img_id, labels_json, img_b64 = img_info.split('\t')

                        img = Base64Utils.b64_str_to_pil(img_b64)
                        w, h = img.size

                        # image data => image file
                        img_file_name = img_id + '.' + img.format
                        Base64Utils.b64_str_to_file(img_b64, os.path.join(image_folder_name, img_file_name))

                        image_path_id = f'{image_folder_name}.zip@{img_file_name}'

                        # image size => meta data file
                        image_meta_data_out.write(f'{image_path_id} {w} {h}\n')

                        # image info => index file
                        img_label_file_name = img_id + '.txt'
                        index_file_out.write(f'{image_path_id} {label_folder_name}.zip@{img_label_file_name}\n')

                        lp = f'File: {tsv_file_name}, Line {line_idx}: '
                        # labels => files
                        with open(os.path.join(label_folder_name, img_label_file_name), 'w') as label_out:
                            labels = json.loads(labels_json)
                            for label in labels:
                                difficulty = label['diff'] if 'diff' in label else 0
                                if difficulty > 0 and not args.difficulty:
                                    continue

                                if labelmap_exists and label['class'] not in label_name_to_idx:
                                    raise Exception(f'{lp}Illegal class {label["class"]}, not in provided labelmap.')

                                if label['class'] not in label_name_to_idx:
                                    label_name_to_idx[label['class']] = len(label_name_to_idx)

                                label_idx = label_name_to_idx[label['class']]
                                box = verify_and_correct_box_or_none(lp, label['rect'], args.format, w, h)
                                if box is None:
                                    continue

                                label_out.write(f'{label_idx} {box[0]} {box[1]} {box[2]} {box[3]}\n')

                        line_idx += 1

    if not labelmap_exists:
        logger.info(f'Write labelmap to {args.labelmap}')
        with open(args.labelmap, 'w') as labelmap_out:
            idx_to_labels = {label_name_to_idx[key]: key for key in label_name_to_idx}
            for i in range(len(idx_to_labels)):
                labelmap_out.write(idx_to_labels[i] + '\n')

    if args.zip:
        logger.info(f'Zip folder "{image_folder_name}".')
        zip_folder(image_folder_name)
        logger.info(f'Zip folder "{label_folder_name}".')
        zip_folder(label_folder_name)


if __name__ == '__main__':
    main()
