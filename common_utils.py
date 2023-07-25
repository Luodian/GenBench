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

import subprocess
import time
from PIL import Image
from collections import defaultdict
import numpy as np
import codecs
import zipfile

from diffusers import StableDiffusionPipeline
import clip
from vision_benchmark.evaluation.metric import get_metric
from vision_benchmark.datasets import SimpleTokenizer, HFPTTokenizer
from vision_benchmark.evaluation import extract_features, extract_text_features, clip_zeroshot_evaluator
from vision_benchmark.config import config, update_config
import torch
import torch.nn.functional as F

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]


import yaml
def configure_meta_info(args=None):
    if (os.path.exists("./vision_benchmark/resources/datasets/vision_datasets_full.json")) and (not args.force_repack):
        return
    else:
        with open("./vision_benchmark/resources/datasets/vision_datasets_full.json", "r") as f:
            vision_dataset_dict = json.load(f)
        import copy
        cur_name = f"{args.dataset_name}_{args.dataset_type}_{args.wandb_tag}_{args.clip_selection}"

        for item in vision_dataset_dict:
            if item["name"] == cur_name:
                logger.info(f"Dataset {cur_name} already exists, please check ./vision_benchmark/resources/datasets/vision_datasets_full.json.")
                return
        
        gen_vision_dataset_dict = copy.deepcopy(vision_dataset_dict)
        found_item = None
        for item in vision_dataset_dict:
            if item["name"] == f"{args.dataset_name}":
                found_item = item
                break

        if args.dataset_name in ["resisc45_clip", "kitti-distance", "hateful-memes"]:
            posfix = "json"
        else:
            posfix = "txt"
        gen_item = copy.deepcopy(found_item)
        gen_item["name"] = cur_name
        gen_item["train"]["index_path"] = f"{args.wandb_tag}_{args.clip_selection}_train.{posfix}"
        gen_item["train"]["files_for_local_usage"] = [f"{args.wandb_tag}_{args.clip_selection}_train.zip"]
        gen_vision_dataset_dict.append(gen_item)
        # vision_benchmark/resources/datasets/cifar-10.yaml
        with open(f"./vision_benchmark/resources/datasets/{args.dataset_name}.yaml", "r") as f:
            dataset_dict = yaml.load(f, Loader=yaml.FullLoader)
            dataset_dict['DATASET']['DATASET'] = f"{cur_name}"

        with open(f"./vision_benchmark/resources/datasets/{cur_name}.yaml", "w") as f:
            yaml.dump(dataset_dict, f, default_flow_style=False, allow_unicode=True)

        with open("./vision_benchmark/resources/datasets/vision_datasets_full.json", "w") as f:
            json.dump(gen_vision_dataset_dict, f)
        logger.info(f"Done, please check the generated dataset meta info at ./vision_benchmark/resources/datasets/vision_datasets_full.json.")

def process_namings(category):
    if type(category) is list:
        category = category[0]
    return category.replace("/", "or").replace(" ", "_").replace(")", "").replace("(", "").replace("'", "").replace(",", "").replace(".", "")


def multi_gpu_launcher(commands: list, workspace_dir: str):
    logger.info("WARNING: using experimental multi_gpu_launcher.")
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x != ""]
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]

    n_gpus = len(available_gpus)
    procs_by_gpu = [None] * n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                logger.info(f"cd {workspace_dir} && CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}")
                new_proc = subprocess.Popen(f"cd {workspace_dir} && CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}", shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_list = []
        self.data_path_list = []
        for image_item in os.listdir(data_path):
            # get file smaller than a threshold, to prevent grid files.
            if image_item.endswith(".png") and os.path.getsize(os.path.join(data_path, image_item)) != 0:
                img = Image.open(os.path.join(data_path, image_item)).convert("RGB")
                self.data_list.append(img)
                self.data_path_list.append(os.path.join(data_path, image_item))
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image = self.data_list[idx]
        image_path = self.data_path_list[idx]
        if self.transform:
            return self.transform(image), image_path
        else:
            return image, image_path

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
# from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        #Nils: Normalize embeddings
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        ## Nils: Add n_views dimension
        features = torch.unsqueeze(features, 1)

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
