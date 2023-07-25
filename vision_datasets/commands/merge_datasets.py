"""
Merge multiple datasets
"""

import pathlib
from pathlib import PurePosixPath

from vision_datasets import DatasetHub, Usages, DatasetInfo, DatasetManifest
from vision_datasets.common.constants import Formats
from vision_datasets.common.util import write_to_json_file_utf8
from .utils import set_up_cmd_logger

logger = set_up_cmd_logger(__name__)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Merge different datasets of the same type into one.')
    parser.add_argument('names', nargs='+', help='names of datasets to be merged.')
    parser.add_argument('-i', '--new_name', type=str, help='name of the merged dataset.', required=True)
    parser.add_argument('-r', '--reg_json', type=pathlib.Path, default=None, help="dataset registration json path.", required=True)
    parser.add_argument('-k', '--blob_container', type=str, help="blob container url.", required=False, default=None)
    parser.add_argument('-f', '--local_dir', type=pathlib.Path, help="local dir for dataet files.", default=pathlib.Path('./'), required=False)

    return parser


def update_dataset_info(target: dict, source: DatasetInfo):
    rt = pathlib.Path(source.root_folder)
    files_for_local_usage = {x: [str(PurePosixPath(rt / z)) for z in y] for x, y in source.files_for_local_usage.items()}

    for k in files_for_local_usage:
        target[k] = target.get(k, {})
        target[k]['files_for_local_usage'] = target[k].get('files_for_local_usage', []) + files_for_local_usage[k]

    if 'type' not in target:
        target['type'] = source.type
    else:
        assert target['type'] == source.type

    return target


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    if len(args.names) <= 1:
        raise ValueError('No datasets to be merged.')

    dataset_resources = DatasetHub(args.reg_json.read_text())
    manifests = []
    merged_dataset_info_dict = {
        'name': args.new_name,
        'description': f'A merged dataset of {",".join(args.names)}.',
        'format': Formats.COCO,
        'root_folder': '',
        'version': 1,
    }

    for name in args.names:
        dataset_info = dataset_resources.dataset_registry.get_dataset_info(name)
        merged_dataset_info_dict = update_dataset_info(merged_dataset_info_dict, dataset_info)

    for phase in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
        manifests.clear()
        for name in args.names:
            manifest = dataset_resources.create_dataset_manifest(args.blob_container, str(args.local_dir.as_posix()), name, usage=phase)
            if not manifest:
                continue
            manifest = manifest[0]

            manifests.append(manifest)

        if not manifests:
            return

        merged_manifest = DatasetManifest.merge(*manifests, flavor=1)
        coco_json = merged_manifest.generate_coco_annotations()
        index_file = f'{merged_dataset_info_dict["name"]}_{phase}.json'
        write_to_json_file_utf8(coco_json, index_file)
        merged_dataset_info_dict[phase]['index_path'] = index_file
        merged_dataset_info_dict[phase]['num_images'] = len(merged_manifest)
        merged_dataset_info_dict['num_classes'] = len(merged_manifest.labelmap)

    write_to_json_file_utf8([merged_dataset_info_dict], f'{args.new_name}_reg.json')


if __name__ == '__main__':
    main()
