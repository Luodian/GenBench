from vision_benchmark.datasets import class_map, template_map
from loguru import logger
import torch
import sys
import os
import shutil
import json
from collections import defaultdict
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from common_utils import process_namings, imagenet_templates_small, CustomDataset
import re

def format_cls_name(class_name):
    if type(class_name) == list:
        class_name = '_and_'.join(class_name)
    new_name = class_name
    new_name = re.sub(r'[^0-9a-zA-Z_]', '_', new_name)
    new_name = re.sub(r'_+', '_', new_name).strip('_')
    return new_name

def get_text_similarity(ref_image_path, dataset_name, ref_shot=5):
    per_category_selected = defaultdict(list)
    concept_list = class_map[dataset_name]
    template_list = template_map[dataset_name]
    parent_output_dir = os.path.dirname(ref_image_path)
    image_path = os.path.join(parent_output_dir, f"images/{dataset_name}")
    meta_info_path = os.path.join(parent_output_dir, f"metas/{dataset_name}")
    ref_shot = ref_shot if ref_shot else 5
    total_sim = 0
    for concept in concept_list:
        saved_concept_dir = format_cls_name(concept)
        with open(os.path.join(meta_info_path, f"{saved_concept_dir}.json"), "r") as f:
            meta_info = json.load(f)
            # logger.info(f"category: {saved_concept_dir}, meta_info: {len(meta_info.keys())}")
            for name in meta_info:
                meta_info[name]["avg_dist"] = np.mean([float(x["dist"]) for x in meta_info[name]["search_meta"]])
            sorted_meta_info = {k: v for k, v in sorted(meta_info.items(), key=lambda x: x[1]["avg_dist"], reverse=True)[:ref_shot]}
            if (sum([float(x["avg_dist"]) for x in sorted_meta_info.values()]) / len(sorted_meta_info)) > 0:
                total_sim += sum([float(x["avg_dist"]) for x in sorted_meta_info.values()]) / len(sorted_meta_info)
            else:
                logger.info(f"category: {saved_concept_dir}, meta_info: {len(meta_info.keys())}")

    logger.info(f"Mean sim for {dataset_name}: {total_sim / len(concept_list):.2f}")

ref_image_path = '/home/v-boli7/azure_storage/data/retrieval_laion400m/images'
ref_shot = 10

dataset_list = [
    "caltech-101",
    "cifar-10",
    "cifar-100",
    "country211",
    "dtd",
    "eurosat_clip",
    "fer-2013",
    "fgvc-aircraft-2013b-variants102",
    "food-101",
    "gtsrb",
    "hateful-memes",
    "kitti-distance",
    "mnist",
    "oxford-flower-102",
    "oxford-iiit-pets",
    "patch-camelyon",
    "rendered-sst2",
    "resisc45_clip",
    "stanford-cars",
    "voc-2007-classification",
]

for dataset_name in dataset_list:
    get_text_similarity(ref_image_path, dataset_name, ref_shot)