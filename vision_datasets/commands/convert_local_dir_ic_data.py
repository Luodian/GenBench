"""
LEGACY: prepare dataset from local dir to iris format
"""

import argparse
import datetime
import pathlib
import os
import zipfile

from vision_datasets import DatasetTypes, Usages
from vision_datasets.common.util import write_to_json_file_utf8


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, _, files in os.walk(path):
        for file in files:
            k = os.path.relpath(os.path.join(root, file), path)
            ziph.write(os.path.join(root, file), k)


def create_argparse():
    parser = argparse.ArgumentParser('Prepare the annotation files, data reg json, and zip files for vision-datasets following iris format.')
    parser.add_argument('--name', type=str, default='Birdsnap', help="Dataset name.")
    parser.add_argument('--type', '-t', type=str, default=DatasetTypes.IC_MULTICLASS, help="type of dataset.", choices=[DatasetTypes.IC_MULTICLASS, DatasetTypes.IC_MULTILABEL])
    parser.add_argument('--description', '-d', type=str, help="Dataset description.", required=True)
    parser.add_argument('--contact', '-c', type=str, help="contact person.", required=False)
    parser.add_argument('--train_folder', '-tr', type=pathlib.Path, default="/home/v-boli7/vlpdataset_storage/SUN397/train", help="Folder including training images.")
    parser.add_argument('--val_folder', '-v', type=pathlib.Path, default="/home/v-boli7/vlpdataset_storage/SUN397/val", help="Folder including validation images.")
    parser.add_argument('--test_folder', '-te', type=pathlib.Path, default="/home/v-boli7/vlpdataset_storage/SUN397/val", help="Folder including test images.")
    return parser


def main():
    parser = create_argparse()
    args = parser.parse_args()

    labelmap_file = 'labels.txt'
    today = datetime.datetime.now().strftime('%Y%m%d')
    reg_json = {
        'name': args.name,
        'description': args.description,
        'contact': args.contact,
        'version': 1,
        "type": args.type,
        "root_folder": f"classification/{args.name.replace('-', '_')}_{today}",
        "labelmap": labelmap_file,
    }

    folder_by_usage = {
        Usages.TRAIN_PURPOSE: args.train_folder,
        Usages.VAL_PURPOSE: args.val_folder,
        Usages.TEST_PURPOSE: args.test_folder
    }

    classes = sorted(os.listdir(folder_by_usage[Usages.TRAIN_PURPOSE]))
    reg_json['num_classes'] = len(classes)
    os.makedirs(f"/home/v-boli7/projects/Genforce/{args.name}", exist_ok=True)
    
    with open(f"{args.name}/{labelmap_file}", 'w') as label_out:
        label_out.write('\n'.join(classes))

    n_images = {usage: 0 for usage in folder_by_usage.keys()}

    image_root_path = str(args.train_folder).replace('train', '')

    for usage, folder in folder_by_usage.items():
        if not folder:
            continue

        with open(f'{args.name}/{usage}.txt', 'w') as index_file:
            for i, c in enumerate(classes):
                for img_file in (folder / c).iterdir():
                    img_path = img_file.as_posix()
                    img_path = str(img_path).replace(f'{folder}/', f'{folder}.zip@')
                    relpath = os.path.relpath(img_path, image_root_path)
                    index_file.write(f'{relpath} {i}\n')
                    n_images[usage] += 1

        with zipfile.ZipFile(f'{args.name}/{usage}.zip', 'w') as zipf:
            zipdir(folder, zipf)

        reg_json[usage] = {
            "index_path": f"{usage}.txt",
            "files_for_local_usage": [
                f"{usage}.zip"
            ],
            "num_images": n_images[usage]
        }

    write_to_json_file_utf8(reg_json, f'{args.name}/reg.json')


if __name__ == '__main__':
    main()
