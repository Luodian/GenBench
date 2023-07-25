import base64
import io
import locale
from loguru import logger
import os
import pathlib
import json
from typing import Union
from vision_datasets import DatasetTypes, DatasetManifest, Usages
from vision_datasets.common.image_loader import PILImageLoader
from vision_datasets.common.util import FileReader
from tqdm import tqdm
import zipfile


def set_up_cmd_logger(name):
    return logger


logger = set_up_cmd_logger(__name__)

TSV_FORMAT_LTRB = 'ltrb'
TSV_FORMAT_LTWH_NORM = 'ltwh-normalized'


class Base64Utils:
    def b64_str_to_pil(img_b64_str: str):
        assert img_b64_str

        return PILImageLoader.load_from_stream(io.BytesIO(base64.b64decode(img_b64_str)))

    def file_to_b64_str(filepath: pathlib.Path):
        assert filepath

        fr = FileReader()
        with fr.open(filepath.as_posix(), "rb") as file_in:
            return base64.b64encode(file_in.read()).decode('utf-8')

    def b64_str_to_file(b64_str: str, file_name: Union[pathlib.Path, str]):
        assert b64_str
        assert file_name

        with open(file_name, 'wb') as file_out:
            file_out.write(base64.b64decode(b64_str))


def add_args_to_locate_dataset_from_name_and_reg_json(parser):
    parser.add_argument('name', type=str, help='Dataset name.')
    parser.add_argument('--reg_json', '-r', type=pathlib.Path, default=None, help='dataset registration json file path.', required=False)
    parser.add_argument('--version', '-v', type=int, help='Dataset version.', default=None)
    parser.add_argument('--usages', '-u', nargs='+', choices=[Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE],
                        help='Usage(s) to check.', default=[Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE])

    parser.add_argument('--blob_container', '-k', type=str, help='Blob container (sas) url', required=False)
    parser.add_argument('--local_dir', '-f', type=pathlib.Path, required=False, help='Check the dataset in this folder. Folder will be created if not exist and blob_container is provided.')


def add_args_to_locate_dataset(parser):
    add_args_to_locate_dataset_from_name_and_reg_json(parser)

    parser.add_argument('--coco_json', '-c', type=pathlib.Path, default=None, help='Single coco json file to check.', required=False)
    parser.add_argument('--data_type', '-t', type=str, default=None, help='Type of data.', choices=DatasetTypes.VALID_TYPES, required=False)


def get_or_generate_data_reg_json_and_usages(args):
    def _generate_reg_json(name, type, coco_path):
        data_info = [
            {
                'name': name,
                'version': 1,
                'type': type,
                'format': 'coco',
                'root_folder': '',
                'train': {
                    'index_path': coco_path.name
                }
            }
        ]

        return json.dumps(data_info)

    if args.reg_json:
        usages = args.usages or [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]
        data_reg_json = args.reg_json.read_text()
    else:
        assert args.coco_json, '--coco_json not provided'
        assert args.data_type, '--data_type not provided'
        usages = [Usages.TRAIN_PURPOSE]
        data_reg_json = _generate_reg_json(args.name, args.data_type, args.coco_json)

    return data_reg_json, usages


def zip_folder(folder_name):
    logger.info(f'zipping {folder_name}.')

    zip_file = zipfile.ZipFile(f'{folder_name}.zip', 'w', zipfile.ZIP_STORED)
    i = 0
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if i % 5000 == 0:
                logger.info(f'zipped {i} images')

            zip_file.write(os.path.join(root, file))
            i += 1

    zip_file.close()


def generate_reg_json(name, type, coco_path):
    data_info = [
        {
            'name': name,
            'version': 1,
            'type': type,
            'format': 'coco',
            'root_folder': '',
            'train': {
                'index_path': coco_path.name
            }
        }
    ]

    return json.dumps(data_info)


def convert_to_tsv(manifest: DatasetManifest, file_path):
    with open(file_path, 'w', encoding='utf-8') as file_out:
        for img in tqdm(manifest.images, desc=f'Writing to {file_path}'):
            converted_labels = []
            for label in img.labels:
                if manifest.data_type in [DatasetTypes.IC_MULTILABEL, DatasetTypes.IC_MULTICLASS]:
                    tag_name = manifest.labelmap[label]
                    converted_label = {'class': tag_name}
                elif manifest.data_type == DatasetTypes.OD:
                    tag_name = manifest.labelmap[label[0]]
                    rect = [int(x) for x in label[1:5]]

                    # to LTRB format
                    converted_label = {'class': tag_name, 'rect': rect}
                elif manifest.data_type == DatasetTypes.IMCAP:
                    converted_label = {'caption': label}

                converted_labels.append(converted_label)

            b64img = Base64Utils.file_to_b64_str(pathlib.Path(img.img_path))
            file_out.write(f'{img.id}\t{json.dumps(converted_labels, ensure_ascii=False)}\t{b64img}\n')


def guess_encoding(tsv_file):
    """guess the encoding of the given file https://stackoverflow.com/a/33981557/
    """
    assert tsv_file

    with io.open(tsv_file, 'rb') as f:
        data = f.read(5)
    if data.startswith(b'\xEF\xBB\xBF'):  # UTF-8 with a "BOM"
        return 'utf-8-sig'
    elif data.startswith(b'\xFF\xFE') or data.startswith(b"\xFE\xFF"):
        return 'utf-16'
    else:  # in Windows, guessing utf-8 doesn't work, so we have to try
        # noinspection PyBroadException
        try:
            with io.open(tsv_file, encoding='utf-8') as f:
                f.read(222222)
                return 'utf-8'
        except Exception:
            return locale.getdefaultlocale()[1]


def verify_and_correct_box_or_none(lp, box, data_format, img_w, img_h):
    error_msg = f'{lp} Illegal box [{", ".join([str(x) for x in box])}], img wxh: {img_w}, {img_h}'
    if len([x for x in box if x < 0]) > 0:
        logger.error(f'{error_msg}. Skip this box.')
        return None

    if data_format == TSV_FORMAT_LTWH_NORM:
        box[2] = int((box[0] + box[2]) * img_w)
        box[3] = int((box[1] + box[3]) * img_h)
        box[0] = int(box[0] * img_w)
        box[1] = int(box[1] * img_h)

    boundary_ratio_limit = 1.02
    if box[0] >= img_w or box[1] >= img_h or box[2] / img_w > boundary_ratio_limit \
            or box[3] / img_h > boundary_ratio_limit or box[0] >= box[2] or box[1] >= box[3]:
        logger.error(f'{error_msg}. Skip this box.')
        return None

    box[2] = min(box[2], img_w)
    box[3] = min(box[3], img_h)

    return box
