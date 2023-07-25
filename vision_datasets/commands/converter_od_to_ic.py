"""
Convert a detection dataset into classification dataset
"""

import argparse
import multiprocessing
import os
import pathlib
import shutil

from vision_datasets import DatasetHub
from vision_datasets.common.manifest_dataset import DetectionAsClassificationByCroppingDataset
from vision_datasets.common.util import write_to_json_file_utf8

from .utils import add_args_to_locate_dataset, get_or_generate_data_reg_json_and_usages, set_up_cmd_logger

logger = set_up_cmd_logger(__name__)


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Convert OD dataset to ic dataset.')
    add_args_to_locate_dataset(parser)
    parser.add_argument('-o', '--output_folder', type=pathlib.Path, required=True, help='target folder of the converted classification dataset')

    parser.add_argument('-zb', '--zoom_ratio_bounds', type=str, required=False,
                        help='lower and bound of the ratio that box height and width can expand (>1) or shrink (0-1), during cropping, e.g, 0.8/1.2')
    parser.add_argument('-sb', '--shift_relative_bounds', type=str, required=False,
                        help='lower/upper bounds of relative ratio wrt box width and height that a box can shift, during cropping, e.g., "-0.3/0.1"')
    parser.add_argument('-np', '--n_copies', type=int, required=False, default=1, help='number of copies per bbox')
    parser.add_argument('-s', '--rnd_seed', type=int, required=False, help='random see for box expansion/shrink/shifting.', default=0)
    parser.add_argument('--zip', dest='zip', action='store_true', help='Flag to add zip prefix to the image paths.')

    return parser


def process_usage(params):
    args, data_reg_json, aug_params, usage = params

    logger.info(f'download dataset manifest for {args.name}...')
    dataset_resources = DatasetHub(data_reg_json)
    dataset = dataset_resources.create_manifest_dataset(args.blob_container, str(args.local_dir.as_posix()), args.name, usage=usage, coordinates='absolute')
    if not dataset:
        logger.info(f'Skipping non-existent phase {usage}.')
        return

    logger.info(f'start conversion for {args.name}...')
    ic_dataset = DetectionAsClassificationByCroppingDataset(dataset, aug_params)
    manifest = ic_dataset.generate_manifest(dir=usage, n_copies=args.n_copies)

    coco = manifest.generate_coco_annotations()
    if args.zip:
        for img in coco['images']:
            img['zip_file'] = f'{usage}.zip'
        write_to_json_file_utf8(coco, args.output_folder / f'{usage}.json')
    shutil.move(f'{usage}', f'{args.output_folder.as_posix()}/', copy_function=shutil.copytree)


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    aug_params = {}
    if args.zoom_ratio_bounds:
        low, up = args.zoom_ratio_bounds.split('/')
        aug_params['zoom_ratio_bounds'] = (float(low), float(up))

    if args.shift_relative_bounds:
        low, up = args.shift_relative_bounds.split('/')
        aug_params['shift_relative_bounds'] = (float(low), float(up))

    if aug_params:
        aug_params['rnd_seed'] = args.rnd_seed

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.blob_container and args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)

    data_reg_json, usages = get_or_generate_data_reg_json_and_usages(args)
    params = [(args, data_reg_json, aug_params, phase) for phase in usages]

    with multiprocessing.Pool(len(usages)) as pool:
        pool.map(process_usage, params)


if __name__ == '__main__':
    main()
