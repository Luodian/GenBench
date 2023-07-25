"""
Convert a dataset or coco json into TSV format
"""

import argparse
import pathlib
from vision_datasets import DatasetRegistry, DatasetHub
from vision_datasets.commands.utils import add_args_to_locate_dataset, convert_to_tsv, get_or_generate_data_reg_json_and_usages, set_up_cmd_logger

logger = set_up_cmd_logger(__name__)


def logging_prefix(dataset_name, version):
    return f'Dataset convert to TSV {dataset_name}, version {version}: '


def main():
    parser = argparse.ArgumentParser('Convert a dataset to TSV(s)')
    add_args_to_locate_dataset(parser)
    parser.add_argument('--output_dir', '-o', type=pathlib.Path, required=False, default=pathlib.Path('./'), help='TSV file(s) will be saved here.')

    args = parser.parse_args()
    prefix = logging_prefix(args.name, args.version)

    data_reg_json, usages = get_or_generate_data_reg_json_and_usages(args)

    dataset_info = DatasetRegistry(data_reg_json).get_dataset_info(args.name, args.version)
    if not dataset_info:
        logger.error(f'{prefix} dataset does not exist.')
        return
    else:
        logger.info(f'{prefix} dataset found in registration file.')

    vision_datasets = DatasetHub(data_reg_json)
    if args.blob_container and args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)

    for usage in usages:
        logger.info(f'{prefix} Check dataset with usage: {usage}.')

        # if args.local_dir is none, then this check will directly try to access data from azure blob. Images must be present in uncompressed folder on azure blob.
        dataset_manifest = vision_datasets.create_dataset_manifest(container_sas=args.blob_container, local_dir=args.local_dir, name=dataset_info.name, version=args.version, usage=usage)
        if dataset_manifest:
            dataset_manifest = dataset_manifest[0]
            convert_to_tsv(dataset_manifest, args.output_dir / f"{usage}.tsv")
        else:
            logger.info(f'{prefix} No split for {usage} available.')


if __name__ == '__main__':
    main()
