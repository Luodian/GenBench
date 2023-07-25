from vision_benchmark.datasets import class_map, template_map
from vision_benchmark.common.constants import get_dataset_hub, VISION_DATASET_STORAGE
from loguru import logger
import torch
import sys
import os
import shutil
import json
import glob
import random

import pickle
import subprocess
import time
from PIL import Image
from collections import defaultdict
import numpy as np
import codecs
import zipfile

from diffusers import StableDiffusionPipeline
import clip
import yaml
import contextlib
from tqdm.auto import tqdm

from vision_benchmark.evaluation.metric import get_metric
from vision_benchmark.datasets import SimpleTokenizer, HFPTTokenizer
from vision_benchmark.evaluation import extract_features, extract_text_features, clip_zeroshot_evaluator
from vision_benchmark.config import config, update_config
import torch
import torch.nn.functional as F
from common_utils import process_namings, imagenet_templates_small, CustomDataset


resisc45_name_map = {
    "airplane": 1,
    "airport": 2,
    "baseball_diamond": 3,
    "basketball_court": 4,
    "beach": 5,
    "bridge": 6,
    "chaparral": 7,
    "church": 8,
    "circular_farmland": 9,
    "cloud": 10,
    "commercial_area": 11,
    "dense_residential": 12,
    "desert": 13,
    "forest": 14,
    "freeway": 15,
    "golf_course": 16,
    "ground_track_field": 17,
    "harbor": 18,
    "industrial_area": 19,
    "intersection": 20,
    "island": 21,
    "lake": 22,
    "meadow": 23,
    "medium_residential": 24,
    "mobile_home_park": 25,
    "mountain": 26,
    "overpass": 27,
    "palace": 28,
    "parking_lot": 29,
    "railway": 30,
    "railway_station": 31,
    "rectangular_farmland": 32,
    "river": 33,
    "roundabout": 34,
    "runway": 35,
    "sea_ice": 36,
    "ship": 37,
    "snowberg": 38,
    "sparse_residential": 39,
    "stadium": 40,
    "storage_tank": 41,
    "tennis_court": 42,
    "terrace": 43,
    "thermal_power_station": 44,
    "wetland": 45,
}

kitti_distance_name_map = {
    "a_photo_i_took_of_a_car_on_my_left_or_right_side": 1,
    "a_photo_i_took_with_a_car_in_the_distance": 2,
    "a_photo_i_took_with_a_car_nearby": 3,
    "a_photo_i_took_with_no_car": 4,
}

hateful_memes_name_map = {
    "hatespeech_meme": 1,
    "meme": 2,
}

def write_json(per_category_selected=None, args=None):
    if args.dataset_name == "resisc45_clip":
        json_pattern = {
            "categories": [
                {"id": 1, "name": "airplane"},
                {"id": 2, "name": "airport"},
                {"id": 3, "name": "baseball_diamond"},
                {"id": 4, "name": "basketball_court"},
                {"id": 5, "name": "beach"},
                {"id": 6, "name": "bridge"},
                {"id": 7, "name": "chaparral"},
                {"id": 8, "name": "church"},
                {"id": 9, "name": "circular_farmland"},
                {"id": 10, "name": "cloud"},
                {"id": 11, "name": "commercial_area"},
                {"id": 12, "name": "dense_residential"},
                {"id": 13, "name": "desert"},
                {"id": 14, "name": "forest"},
                {"id": 15, "name": "freeway"},
                {"id": 16, "name": "golf_course"},
                {"id": 17, "name": "ground_track_field"},
                {"id": 18, "name": "harbor"},
                {"id": 19, "name": "industrial_area"},
                {"id": 20, "name": "intersection"},
                {"id": 21, "name": "island"},
                {"id": 22, "name": "lake"},
                {"id": 23, "name": "meadow"},
                {"id": 24, "name": "medium_residential"},
                {"id": 25, "name": "mobile_home_park"},
                {"id": 26, "name": "mountain"},
                {"id": 27, "name": "overpass"},
                {"id": 28, "name": "palace"},
                {"id": 29, "name": "parking_lot"},
                {"id": 30, "name": "railway"},
                {"id": 31, "name": "railway_station"},
                {"id": 32, "name": "rectangular_farmland"},
                {"id": 33, "name": "river"},
                {"id": 34, "name": "roundabout"},
                {"id": 35, "name": "runway"},
                {"id": 36, "name": "sea_ice"},
                {"id": 37, "name": "ship"},
                {"id": 38, "name": "snowberg"},
                {"id": 39, "name": "sparse_residential"},
                {"id": 40, "name": "stadium"},
                {"id": 41, "name": "storage_tank"},
                {"id": 42, "name": "tennis_court"},
                {"id": 43, "name": "terrace"},
                {"id": 44, "name": "thermal_power_station"},
                {"id": 45, "name": "wetland"},
            ],
            "images": [],
            "annotations": [],
        }
        convert_map = resisc45_name_map
    elif args.dataset_name == "kitti-distance":
        json_pattern = {
            "categories": [
                {"id": 1, "name": "a photo i took of a car on my left or right side."},
                {"id": 2, "name": "a photo i took with a car nearby."},
                {"id": 3, "name": "a photo i took with a car in the distance."},
                {"id": 4, "name": "a photo i took with no car."},
            ],
            "images": [],
            "annotations": [],
        }
        convert_map = kitti_distance_name_map
    elif args.dataset_name == "hateful-memes":
        json_pattern = {
            "categories": [{"id": 1, "name": "non-hateful"}, {"id": 2, "name": "hateful"}],
            "images": [],
            "annotations": [],
        }
        convert_map = hateful_memes_name_map
    id_count = 1
    width = 512
    height = 512

    for cate_name in per_category_selected:
        img_item_list = per_category_selected[cate_name]
        for img_item in img_item_list:
            image_item = {}
            annotation_item = {}
            annotation_item["id"] = id_count
            annotation_item["image_id"] = id_count
            annotation_item["category_id"] = convert_map[cate_name]
            image_item["id"] = id_count
            image_item["height"] = height
            image_item["width"] = width
            image_item["file_name"] = f"{args.wandb_tag}_{args.clip_selection}_train.zip@{cate_name}/{img_item}"
            json_pattern["images"].append(image_item)
            json_pattern["annotations"].append(annotation_item)
            id_count += 1

    with codecs.open(f"{args.output_dir}/{args.wandb_tag}_{args.clip_selection}_train.json", "w", encoding="utf-8") as f:
        json.dump(json_pattern, f, ensure_ascii=False)

    shutil.copy(os.path.join(args.output_dir, f"{args.wandb_tag}_{args.clip_selection}_train.json"), os.path.join(args.elevator_dataset_full_path, f"{args.wandb_tag}_{args.clip_selection}_train.json"))


def write_txt(class_names, per_category_selected=None, args=None):
    convert_maps = {}
    line_count = 0
    for item in class_names:
        convert_maps[process_namings(item)] = line_count
        line_count += 1

    # if exsits the txt file, remove it
    if os.path.exists(f"{args.output_dir}/{args.wandb_tag}_{args.clip_selection}_train.txt"):
        os.remove(f"{args.output_dir}/{args.wandb_tag}_{args.clip_selection}_train.txt")

    with open(f"{args.output_dir}/{args.wandb_tag}_{args.clip_selection}_train.txt", "w") as f:
        for name in per_category_selected:
            labelid = convert_maps[name]
            for img_item in per_category_selected[name]:
                f.writelines(f"{args.wandb_tag}_{args.clip_selection}_train.zip@{name}/{img_item} {labelid}\n")

    shutil.copy(os.path.join(args.output_dir, f"{args.wandb_tag}_{args.clip_selection}_train.txt"), os.path.join(args.elevator_dataset_full_path, f"{args.wandb_tag}_{args.clip_selection}_train.txt"))
    
import glob
def packing_generated_images(class_names, args=None):
    logger.info(f"Start packing generated images for {args.dataset_name} ...")
    # if os.path.exists(os.path.join(args.output_dir, "cleaned.txt")) is False and args.dataset_type == "generated":
    # # if dataset_type == "generated":
    #     for name in os.listdir(args.output_dir):
    #         if os.path.isdir(os.path.join(args.output_dir, name)):
    #             logger.info(f"cleaning {name} ...")
    #             if os.path.exists(os.path.join(args.output_dir, name, "samples")):
    #                 logger.info(f"move {os.path.join(args.output_dir, name, 'samples')}")
    #                 os.system(f"cd {os.path.join(args.output_dir, name)} && mv samples/* ./")
    #                 # shutil.move(os.path.join(work_dir, name, 'samples'), os.path.join(work_dir, name))
    #                 shutil.rmtree(os.path.join(args.output_dir, name, "samples"))

    #             # clean a huge grid file
    #             grid_files = glob.glob(os.path.join(args.output_dir, name, "grid-*.png"))
    #             for grid_file in grid_files:
    #                 logger.info(f"remove {grid_file}")
    #                 os.remove(os.path.join(args.output_dir, name, grid_file))

    #             text_files = glob.glob(os.path.join(args.output_dir, name, "*.txt"))
    #             for txt_file in text_files:
    #                 logger.info(f"remove {txt_file}")
    #                 os.remove(os.path.join(args.output_dir, name, grid_file))

    #             os.system(f'cd {os.path.join(output_dir, name)} && find .  | grep \'png\' | nl -nrz -w4 -v1 | while read n f; do mv "$f" "$n.png"; done')

    #     open(os.path.join(output_dir, "cleaned.txt"), "w").close()

    if args.force_repack:
        # zip images with high CLIP score
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/16", device=device)
        per_category_selected = defaultdict(list)
        for category in class_names: # align with the naming convention of the generated images, process the tedious namings.
            saved_concept_dir = process_namings(category)
            category_dataset = CustomDataset(os.path.join(args.output_dir, saved_concept_dir), transform=preprocess)
            category_loader = torch.utils.data.DataLoader(category_dataset, batch_size=128, shuffle=False, num_workers=16)
            text = clip.tokenize(f"a photo of a {category}").to(device)
            for data, data_path in tqdm(category_loader, ascii=True, desc=f"Processing {saved_concept_dir}"):
                data = data.to(device)
                with torch.no_grad():
                    logits_per_image, logits_per_text = model(data, text)
                    probs = logits_per_image.cpu().numpy()
                    for i, prob in enumerate(probs):
                        per_category_selected[saved_concept_dir].append((f"{data_path[i].split('/')[-1]}", float(prob)))
                        
        with open(os.path.join(args.output_dir, f"{args.dataset_name}_{args.dataset_type}_clip_score.json"), "w") as f:
            json.dump(per_category_selected, f)
    else:
        with open(os.path.join(args.output_dir, f"{args.dataset_name}_{args.dataset_type}_clip_score.json"), "r") as f:
            per_category_selected = json.load(f)
            
    # clip selection
    for category in class_names: # align with the naming convention of the generated images, process the tedious namings.
        saved_concept_dir = process_namings(category)
        if args.clip_selection > 0:
            logger.info(f"Selecting {args.clip_selection} images with highest CLIP score for {saved_concept_dir} ...")
            per_category_selected[saved_concept_dir] = [item[0] for item in sorted(per_category_selected[saved_concept_dir], key=lambda x: x[1], reverse=True)[: args.clip_selection]]
        else:
            logger.info(f"Selecting {args.random_selection_num} images at random for {saved_concept_dir} ...")
            per_category_selected[saved_concept_dir] = random.sample([item[0] for item in per_category_selected[saved_concept_dir]], args.random_selection_num)

    with zipfile.ZipFile(os.path.join(args.output_dir, f"{args.wandb_tag}_{args.clip_selection}_train.zip"), "w", zipfile.ZIP_DEFLATED) as zipf:
        for category in per_category_selected:
            logger.info(f"processing category: {category}")
            for name in per_category_selected[category]:
                logger.info(f"=> processing image: {name}")
                zipf.write(os.path.join(args.output_dir, category, name), f"{category}/{name}")

    if args.dataset_name in ["resisc45_clip", "kitti-distance", "hateful-memes"]:
        write_json(per_category_selected=per_category_selected, args=args)
    else:
        write_txt(class_names=class_names, per_category_selected=per_category_selected, args=args)
        
    # dataset_hub = get_dataset_hub()
    # from vision_datasets import Usages
    # manifest = dataset_hub.create_dataset_manifest(VISION_DATASET_STORAGE, local_dir=f"{args.elevator_root_path}/datasets/", name=args.dataset_name, usage=Usages.TEST_PURPOSE)
    # if manifest:
    #     elevator_dataset_full_path = os.path.join(args.elevator_root_path, "datasets", manifest[1].root_folder)
    shutil.copy(os.path.join(args.output_dir, f"{args.wandb_tag}_{args.clip_selection}_train.zip"), os.path.join(args.elevator_dataset_full_path, f"{args.wandb_tag}_{args.clip_selection}_train.zip"))
    open(os.path.join(args.output_dir, "packed.txt"), "w").close()


def packing_retrieval_images(class_names, args=None):
    per_category_limit = args.clip_selection if args.clip_selection > 0 else args.random_selection_num
    per_category_selected = defaultdict(list)
    parent_output_dir = os.path.dirname(os.path.dirname(args.output_dir))
    image_path = os.path.join(parent_output_dir, f"images/{args.dataset_name}")
    meta_info_path = os.path.join(parent_output_dir, f"metas/{args.dataset_name}")
    logger.info(f"{args.dataset_name} per category limit: {per_category_limit}")

    for category in class_names:
        saved_category_dir = process_namings(category)
        with open(os.path.join(meta_info_path, f"{saved_category_dir}.json"), "r") as f:
            meta_info = json.load(f)
            logger.info(f"category: {saved_category_dir}, meta_info: {len(meta_info.keys())}")
            for name in meta_info:
                meta_info[name]["avg_dist"] = np.mean([float(x["dist"]) for x in meta_info[name]["search_meta"]])
            sorted_meta_info = {k: v for k, v in sorted(meta_info.items(), key=lambda x: x[1]["avg_dist"], reverse=True)}
            per_category_selected[saved_category_dir] = list(sorted_meta_info.keys())[:per_category_limit]
            logger.info(f"category: {saved_category_dir}, selected: {len(per_category_selected[saved_category_dir])}")

    with zipfile.ZipFile(os.path.join(image_path, f"{args.wandb_tag}_{args.clip_selection}_train.zip"), "w", zipfile.ZIP_DEFLATED) as zipf:
        for category in per_category_selected:
            logger.info(f"processing category: {category}")
            for name in per_category_selected[category]:
                logger.info(f"\tprocessing image: {name}")
                zipf.write(os.path.join(image_path, category, name), f"{category}/{name}")

    shutil.copy(os.path.join(args.output_dir, f"{args.wandb_tag}_{args.clip_selection}_train.zip"), os.path.join(args.elevator_dataset_full_path, f"{args.wandb_tag}_{args.clip_selection}_train.zip"))
    if args.dataset_name in ["resisc45_clip", "kitti-distance", "hateful-memes"]:
        write_json(per_category_selected=per_category_selected, args=args)
    else:
        write_txt(class_names=class_names, per_category_selected=per_category_selected, args=args)