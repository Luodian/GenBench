"""
Linear classifier implemented with Pytorch Linear class
"""

import gc
import json
from loguru import logger
import sys
import os
import pdb
import pickle
import random
import sys
import time
from re import L
import datetime
import nltk
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from torch import nn
from tqdm import tqdm
from vision_benchmark.datasets import HFPTTokenizer, SimpleTokenizer
from vision_benchmark.evaluation import clip_zeroshot_evaluator, construct_dataloader
from vision_datasets import ManifestDataset
import wandb

from ..common.constants import VISION_DATASET_STORAGE, get_dataset_hub
from ..datasets import class_map, template_map, dataset_metrics
from ..evaluation.metric import get_metric
from ..models import *
from ..optim import build_optimizer
from .feature import FeatureData, create_dataloader, extract_text_features, get_model

nltk.download("punkt")
nltk.download("wordnet")


MULTILABEL_DATASETS = {"voc-2007-classification", "chestx-ray8"}


def gpu_gc():
    gc.collect()
    torch.cuda.empty_cache()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class IC_MultiTask_Classifier(torch.nn.Module):
    """
    Linear classifier.
    """

    def __init__(self, config, l2_lambda, task_list):
        super(IC_MultiTask_Classifier, self).__init__()

        self.backbone = get_model(config, feature_type="image")

        if config.MODEL.NAME.startswith("vit_"):
            self.backbone.head = self.backbone.head_dist = None

        for name, param in self.backbone.named_parameters():
            if (
                name.startswith("text")
                or name.startswith("transformer")
                or name.startswith("token_embedding")
                or name.startswith("ln_final")
                or name.startswith("positional_embedding")
                or name.startswith("logit_scale")
            ):
                param.requires_grad = False

            if config.TRAIN.FREEZE_IMAGE_BACKBONE:
                # freeze for {supervised ViT, MAE, MoCov3} under linear probing settings
                for model_keyword in ["vit", "mae", "mocov3"]:
                    if config.MODEL.NAME.startswith(f"{model_keyword}_"):
                        param.requires_grad = False

                if name.startswith("visual.conv1") or name.startswith("visual.ln_pre") or name.startswith("visual.transformer") or name.startswith("visual"):
                    param.requires_grad = False

        self.input_dim = config.MODEL.SPEC.EMBED_DIM
        self.optim = None
        self.l2_lambda = l2_lambda
        self.channel_bn = torch.nn.BatchNorm1d(
            self.input_dim,
            affine=False,
        )

        if config.TRAIN.MULTI_TASK_HEAD == "shared":
            # create a classifier to handle every task.
            output_dim = sum([len(class_map[task]) for task in task_list])
            self.layers = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, output_dim)])
        elif config.TRAIN.MULTI_TASK_HEAD == "split":
            self.layers = nn.ModuleList()
            for t_name in task_list:
                self.output_dim = len(class_map[t_name])
                self.layer = torch.nn.Linear(self.input_dim, self.output_dim)
                self.layers.append(self.layer)

        if config.TRAIN.INIT_HEAD_WITH_TEXT_ENCODER:
            if config.MODEL.SPEC.TEXT.TOKENIZER == "clip":
                tokenizer = SimpleTokenizer()
            elif "hf_" in config.MODEL.SPEC.TEXT.TOKENIZER:
                tokenizer = HFPTTokenizer(pt_name=config.MODEL.SPEC.TEXT.TOKENIZER[3:])
            else:
                tokenizer = None

            label_embedding_list = []
            zeroshot_weights_list = []
            logger.info("Initializing head with text encoder weights for each task...")

            class_name_list = []
            for dataset_name in task_list:
                if os.path.exists(f"./zeroshot_weights/{dataset_name}.pt"):
                    logger.info(f"load text encoder weights from {dataset_name}.pt")
                    zeroshot_weights = torch.load(f"./zeroshot_weights/{dataset_name}.pt")
                else:
                    zeroshot_weights = extract_text_features(config, dataset_name, tokenizer, model=self.backbone, return_numpy=False)
                    torch.save(zeroshot_weights, f"./zeroshot_weights/{dataset_name}.pt")
                class_name_list.extend(class_map[dataset_name])
                label_embedding_list.append(zeroshot_weights.T)
                zeroshot_weights_list.append(zeroshot_weights)

            # plt.figure(figsize=(10,10), dpi=300)
            # experimentally calculate the label similarity between all tasks
            label_embedding = torch.cat(label_embedding_list, dim=0)
            # 1151 x 1151
            embed_dist = torch.nn.functional.cosine_similarity(label_embedding[:, :, None], label_embedding.t()[None, :, :])
            torch.save(embed_dist, f"./zeroshot_weights/embed_dist.pt")
            # plot using a color palette
            # sns.heatmap(embed_dist.cpu().numpy(), cmap="YlGnBu")
            # # plt.xlabel(class_name_list)
            # # plt.ylabel(class_name_list)
            # cls_ticks = [i for i in range(len(class_name_list))]
            # plt.xticks(ticks=cls_ticks, labels=class_name_list, rotation=90, fontsize=4)
            # plt.yticks(ticks=cls_ticks, labels=class_name_list, fontsize=4)
            # plt.savefig("/mnt/lustre/bli/projects/Elevater_Toolkit_IC/vision_benchmark/label_similarity_caltech101_cifar10.png")
            # plt.close()

            if config.TRAIN.MULTI_TASK_HEAD == "shared":
                start_index = 0
                end_index = 0
                for indx, task in enumerate(task_list):
                    start_index = end_index
                    end_index += len(class_map[task])
                    self.layers[0].weight.data[start_index:end_index] = (zeroshot_weights_list[indx].T.to(self.layers[0].weight.dtype).to(self.layers[0].weight.device).contiguous())
                    self.layers[0].bias.data.fill_(0.0)
            elif config.TRAIN.MULTI_TASK_HEAD == "split":
                for indx in range(len(self.layers)):
                    self.layers[indx].weight.data = (
                        zeroshot_weights_list[indx].T.to(self.layers[indx].weight.dtype).to(self.layers[indx].weight.device).contiguous()
                    )
                    self.layers[indx].bias.data.fill_(0.0)

        # TODO: whether we need them?
        # if config.TRAIN.MERGE_ENCODER_AND_HEAD_PROJ and self.backbone.visual.proj is not None:
        #     encoder_proj = self.backbone.visual.proj
        #     head_proj = self.layers[0].weight.data
        #     head_bias = self.layers[0].bias.data
        #     self.backbone.visual.proj = None
        #     encoder_ic, encoder_oc = encoder_proj.shape
        #     self.channel_bn = torch.nn.BatchNorm1d(
        #         encoder_ic,
        #         affine=False,
        #     )
        #     self.layers = torch.nn.Sequential(torch.nn.Linear(encoder_ic, output_dim))
        #     self.layers[0].weight.data = head_proj @ encoder_proj.T.to(head_proj.dtype).to(head_proj.device)
        #     self.layers[0].bias.data = head_bias

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.logit_scale.requires_grad = config.TRAIN.TRAINABLE_LOGIT_SCALE
        if config.TRAIN.LOGIT_SCALE_INIT == "pretrained":
            self.logit_scale.data = self.backbone.logit_scale.data.to(self.logit_scale.dtype).to(self.logit_scale.device)
        elif config.TRAIN.LOGIT_SCALE_INIT == "ln_cls":
            self.logit_scale.data *= np.log(np.log(config.DATASET.NUM_CLASSES))
        elif config.TRAIN.LOGIT_SCALE_INIT == "clip":
            self.logit_scale.data *= np.log(1 / 0.07)
        else:
            self.logit_scale.data *= 0

        self.normalize_visual_output = config.TRAIN.NORMALIZE_VISUAL_FEATURE

        if not config.TRAIN.USE_CHANNEL_BN:
            self.channel_bn = nn.Identity()

    def forward(self, img, task_index=0):
        pdtype = img.dtype
        feature = self.backbone(img).to(pdtype)
        outputs = self.channel_bn(feature)

        if self.normalize_visual_output:
            outputs = F.normalize(outputs)

        outputs = self.logit_scale.exp() * self.layers[task_index](outputs)
        return outputs


class Classifier(torch.nn.Module):
    """
    Linear classifier.
    """

    def __init__(self, config, l2_lambda):
        super(Classifier, self).__init__()

        self.backbone = get_model(config, feature_type="image")

        if config.MODEL.NAME.startswith("vit_"):
            self.backbone.head = self.backbone.head_dist = None

        for name, param in self.backbone.named_parameters():
            if (
                name.startswith("text")
                or name.startswith("transformer")
                or name.startswith("token_embedding")
                or name.startswith("ln_final")
                or name.startswith("positional_embedding")
                or name.startswith("logit_scale")
            ):
                param.requires_grad = False

            if config.TRAIN.FREEZE_IMAGE_BACKBONE:
                # freeze for {supervised ViT, MAE, MoCov3} under linear probing settings
                for model_keyword in ["vit", "mae", "mocov3"]:
                    if config.MODEL.NAME.startswith(f"{model_keyword}_"):
                        param.requires_grad = False

                if name.startswith("visual.conv1") or name.startswith("visual.ln_pre") or name.startswith("visual.transformer") or name.startswith("visual"):
                    param.requires_grad = False

        input_dim, output_dim = config.MODEL.SPEC.EMBED_DIM, config.DATASET.NUM_CLASSES
        self.optim = None
        self.l2_lambda = l2_lambda
        self.channel_bn = torch.nn.BatchNorm1d(
            input_dim,
            affine=False,
        )
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))

        if config.TRAIN.INIT_HEAD_WITH_TEXT_ENCODER:
            if config.MODEL.SPEC.TEXT.TOKENIZER == "clip":
                tokenizer = SimpleTokenizer()
            elif "hf_" in config.MODEL.SPEC.TEXT.TOKENIZER:
                tokenizer = HFPTTokenizer(pt_name=config.MODEL.SPEC.TEXT.TOKENIZER[3:])
            else:
                tokenizer = None

            zeroshot_weights = extract_text_features(config=config, dataset_name=config.DATASET.DATASET ,tokenizer=tokenizer, model=self.backbone, return_numpy=False)
            self.layers[0].weight.data = zeroshot_weights.T.to(self.layers[0].weight.dtype).to(self.layers[0].weight.device).contiguous()
            self.layers[0].bias.data.fill_(0.0)

        if config.TRAIN.MERGE_ENCODER_AND_HEAD_PROJ and self.backbone.visual.proj is not None:
            encoder_proj = self.backbone.visual.proj
            head_proj = self.layers[0].weight.data
            head_bias = self.layers[0].bias.data
            self.backbone.visual.proj = None
            encoder_ic, encoder_oc = encoder_proj.shape
            self.channel_bn = torch.nn.BatchNorm1d(
                encoder_ic,
                affine=False,
            )
            self.layers = torch.nn.Sequential(torch.nn.Linear(encoder_ic, output_dim))
            self.layers[0].weight.data = head_proj @ encoder_proj.T.to(head_proj.dtype).to(head_proj.device)
            self.layers[0].bias.data = head_bias

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.logit_scale.requires_grad = config.TRAIN.TRAINABLE_LOGIT_SCALE
        if config.TRAIN.LOGIT_SCALE_INIT == "pretrained":
            self.logit_scale.data = self.backbone.logit_scale.data.to(self.logit_scale.dtype).to(self.logit_scale.device)
        elif config.TRAIN.LOGIT_SCALE_INIT == "ln_cls":
            self.logit_scale.data *= np.log(np.log(config.DATASET.NUM_CLASSES))
        elif config.TRAIN.LOGIT_SCALE_INIT == "clip":
            self.logit_scale.data *= np.log(1 / 0.07)
        else:
            self.logit_scale.data *= 0

        self.normalize_visual_output = config.TRAIN.NORMALIZE_VISUAL_FEATURE

        if not config.TRAIN.USE_CHANNEL_BN:
            self.channel_bn = nn.Identity()

    def forward(self, img):
        pdtype = img.dtype
        feature = self.backbone(img).to(pdtype)
        outputs = self.channel_bn(feature)

        if self.normalize_visual_output:
            outputs = F.normalize(outputs)

        outputs = self.logit_scale.exp() * self.layers(outputs)
        return outputs


def multi_task_hyperparameter_sweep(train_dataloader, val_dataloader, task_list, config):
    logger.info(f"=> Learning rate {config.TRAIN.LR}: tuning l2 regularization strength.")
    start = time.time()
    l2_lambda_list = np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=97).tolist()
    l2_lambda_init_idx = [
        i for i, val in enumerate(l2_lambda_list) if val in set(np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=7))
    ]
    peak_idx = -1
    peak_score = 0
    iter_num = 0
    for idx in l2_lambda_init_idx:
        config.defrost()
        config.TRAIN.WD = l2_lambda_list[idx]

        try:
            best_score_list = IC_Multitask_train_task(train_dataloader, val_dataloader, task_list, config, sweep_run=True)
            best_score_ = sum(best_score_list) / len(best_score_list)
        except Exception as e:
            best_score_ = 0.0
            gpu_gc()
            print(e)

        if best_score_ > peak_score:
            peak_idx = idx
            peak_score = best_score_
    logger.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")
    step_span = 8
    while step_span > 0:
        left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list) - 1)
        search_idx = []
        if left != peak_idx:
            search_idx.append(left)
        if right != peak_idx:
            search_idx.append(right)
        for idx in search_idx:
            # WD_SEARCH_LEFT is used in the inital release, whereas we later find WD_SEARCH_IDX to be more stable.
            if config.TRAIN.WD_SEARCH_LEFT:
                config.TRAIN.WD = l2_lambda_list[left]
            else:
                config.TRAIN.WD = l2_lambda_list[idx]

            try:
                best_score_list = IC_Multitask_train_task(train_dataloader, val_dataloader, task_list, config, sweep_run=True)
                best_score_ = sum(best_score_list) / len(best_score_list)
            except Exception as e:
                best_score_ = 0.0
                gpu_gc()
                print(e)

            if best_score_ > peak_score:
                peak_idx = idx
                peak_score = best_score_

        iter_num += 1
        logger.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")
        step_span //= 2

    logger.info(f"=> Learning rate {config.TRAIN.LR}: The best l2 lambda is {l2_lambda_list[peak_idx]}")
    logger.info("=> Learning rate {}: l2 regularization strength tuning duration time: {:.2f}s".format(config.TRAIN.LR, time.time() - start))
    return l2_lambda_list[peak_idx], peak_score


def hyperparameter_sweep(train_dataloader, val_dataloader, config):
    logger.info(f"=> Learning rate {config.TRAIN.LR}: tuning l2 regularization strength.")
    start = time.time()
    l2_lambda_list = np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=97).tolist()
    l2_lambda_init_idx = [
        i for i, val in enumerate(l2_lambda_list) if val in set(np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=7))
    ]
    peak_idx = -1
    peak_score = 0
    iter_num = 0
    for idx in l2_lambda_init_idx:
        config.defrost()
        config.TRAIN.WD = l2_lambda_list[idx]
        try:
            best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
        except:
            best_score_ = 0.0
            gpu_gc()
            continue
        if best_score_ > peak_score:
            peak_idx = idx
            peak_score = best_score_
    logger.info(f"After search lambda interval of {len(l2_lambda_init_idx)} iters: peak l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}, go to fine search.")

    step_span = 8
    while step_span > 0:
        left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list) - 1)
        search_idx = []
        if left != peak_idx:
            search_idx.append(left)
        if right != peak_idx:
            search_idx.append(right)
        for idx in search_idx:
            # WD_SEARCH_LEFT is used in the inital release, whereas we later find WD_SEARCH_IDX to be more stable.
            if config.TRAIN.WD_SEARCH_LEFT:
                config.TRAIN.WD = l2_lambda_list[left]
            else:
                config.TRAIN.WD = l2_lambda_list[idx]

            try:
                best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
            except:
                best_score_ = 0.0
                gpu_gc()
                continue

            if best_score_ > peak_score:
                peak_idx = idx
                peak_score = best_score_
        iter_num += 1
        # logger.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")
        step_span //= 2

    logger.info(f"=> Learning rate {config.TRAIN.LR}, peak l2 lambda is {l2_lambda_list[peak_idx]}, duration time: {time.time() - start:.3f}")
    # logger.info("=> Learning rate {}: l2 regularization strength tuning duration time: {:.2f}s".format(config.TRAIN.LR, )
    return l2_lambda_list[peak_idx], peak_score


def clone_loader(loader, shuffle=True):

    return create_dataloader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


import random


def IC_Multitask_train_task(train_dataloader, test_dataloader, task_list, config, sweep_run=False):
    best_acc1 = 0

    # model = Classifier(config, 0)
    # determine to create a shared/split head model
    model = IC_MultiTask_Classifier(config, 0, task_list)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if sweep_run == False:
        logger.info(f"Number of trainable params: {pytorch_total_params / 1000000}M.")

    # TODO: why need clone?
    # train_dataloader = clone_loader(train_dataloader)

    gpu = config.GPUS

    if len(gpu) == 1:
        torch.cuda.set_device(gpu[0])
        model = model.cuda(gpu[0])

    # define loss function (criterion) and optimizer
    # TODO: need to specifically consider this loss function for voc-2007

    optimizer = build_optimizer(config, model)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC

    # Generate model statistics
    model_info = {}
    visual_backbone = model.backbone.visual if hasattr(model.backbone, "visual") and model.backbone.visual is not None else model.backbone
    model_info["n_trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info["n_visual_params"] = sum(p.numel() for p in visual_backbone.parameters())
    model_info["n_backbone_params"] = sum(p.numel() for p in model.backbone.parameters())
    model_info["n_params"] = sum(p.numel() for p in model.parameters())
    model_info["best_logits"] = [None for _ in range(len(task_list))]

    if sweep_run == False:
        logger.info(f"Train on tasks: {task_list}")
        logger.info(f"{config.TRAIN.MULTI_TASK_HEAD} head is used.")
    
    task_num = len(task_list)
    # progressively flush previous values
    best_acc_list = [0 for i in range(task_num)]
    task_index_list = [i for i in range(task_num)]
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        adjust_learning_rate(optimizer, epoch, config)

        random.shuffle(task_index_list)
        # train for one epoch
        if config.TRAIN.MULTI_TASK_HEAD == "shared":
            criterion = torch.nn.CrossEntropyLoss().cuda(gpu)
            metric_name, metric_result, loss_avg = train_one(train_dataloader, model, criterion, optimizer, epoch, config)
            if sweep_run == False:
                logger.info(f"[Epoch {epoch}]: Train {metric_name} {metric_result:.4f} Loss {loss_avg:.4f}")
        elif config.TRAIN.MULTI_TASK_HEAD == "split":
            for t in task_index_list:
                if task_list[t] in MULTILABEL_DATASETS:
                    criterion = torch.nn.BCEWithLogitsLoss().cuda(gpu)
                else:
                    criterion = torch.nn.CrossEntropyLoss().cuda(gpu)
                metric_name, metric_result, loss_avg = IC_multi_task_train_one(
                    train_loader=train_dataloader[t],
                    model=model,
                    task_index=t,
                    ds_name=task_list[t],
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                )
                if sweep_run == False:
                    logger.info(f"[Epoch {epoch}][Task {task_list[t]}] Train {metric_name}: {metric_result:.3f}")

        # evaluate on each task
        for t in task_index_list:
            metric_name, metric_result, logits = IC_multi_task_validate(
                val_loader=test_dataloader[t],
                model=model,
                task_index=t,
                task_list=task_list,
                criterion=criterion,
                epoch=epoch,
                config=config,
                return_logits=True,
            )
            if sweep_run == False:
                logger.info(f"[Epoch {epoch}][Task {task_list[t]}] Val {metric_name}: {metric_result:.3f}")
            # remember best acc@1 and save checkpoint
            if metric_result > best_acc_list[t]:
                model_info["best_logits"][t] = logits

            best_acc_list[t] = max(metric_result, best_acc_list[t])
            # turn on wandb log only when training, not in hparam searching.
            if sweep_run is False:
                wandb.log(
                    {
                        f"task_{task_list[t]}/epoch": epoch,
                        f"task_{task_list[t]}/train_loss": loss_avg,
                        f"task_{task_list[t]}/train_acc": metric_result,
                        f"task_{task_list[t]}/val_acc": metric_result,
                    }
                )
        avg_best_acc = sum(best_acc_list) / task_num
        if sweep_run == False:
            logger.info(f"[Epoch {epoch}] Avg best acc: {avg_best_acc:.3f}")
    # turn on wandb log only when training, not in hparam searching.
    if sweep_run is False:
        val_data = [[] for i in range(task_num)]
        for t in range(task_num):
            logger.info(f"Task: {task_list[t]} => LR {config.TRAIN.LR}, L2 lambda {config.TRAIN.WD}: Best score: Acc@1 {best_acc_list[t]:.3f}")
            val_data[t] = [task_list[t], best_acc_list[t]]
        val_table = wandb.Table(columns=["Task", "Best Val Acc"], data=val_data)
        val_table.add_data("Average", avg_best_acc)
        wandb.log({"val_table": val_table})

    del model, criterion, optimizer
    gpu_gc()

    if sweep_run:
        return best_acc_list
    else:
        return best_acc_list, model_info


def train_task(train_dataloader, test_dataloader, config, sweep_run=False):
    best_acc1 = 0

    model = Classifier(config, 0)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if sweep_run == False:
        logger.info(f"Number of trainable params: {pytorch_total_params / 1000000}M.")

    train_dataloader = clone_loader(train_dataloader)

    gpu = config.GPUS

    if len(gpu) == 1:
        torch.cuda.set_device(gpu[0])
        model = model.cuda(gpu[0])

    # define loss function (criterion) and optimizer
    if config.DATASET.DATASET in MULTILABEL_DATASETS:
        criterion = torch.nn.BCEWithLogitsLoss().cuda(gpu)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda(gpu)

    optimizer = build_optimizer(config, model)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC

    # Generate model statistics
    model_info = {}
    visual_backbone = model.backbone.visual if hasattr(model.backbone, "visual") and model.backbone.visual is not None else model.backbone
    model_info["n_trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info["n_visual_params"] = sum(p.numel() for p in visual_backbone.parameters())
    model_info["n_backbone_params"] = sum(p.numel() for p in model.backbone.parameters())
    model_info["n_params"] = sum(p.numel() for p in model.parameters())

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        adjust_learning_rate(optimizer, epoch, config)
        # train for one epoch
        train_one(train_dataloader, model, criterion, optimizer, epoch, config)
        # evaluate on validation set
        metric_name, metric_result, logits = validate(test_dataloader, model, criterion, epoch, config, return_logits=True)
        if sweep_run == False:
            logger.info(f"[Epoch {epoch}] Val {metric_name}: {metric_result:.3f}")
        # remember best acc@1 and save checkpoint
        if metric_result > best_acc1:
            model_info["best_logits"] = logits
        best_acc1 = max(metric_result, best_acc1)

    # if sweep_run == False:
    #     if config.TRAIN.SCHEDULE_TYPE == 2 and config.TRAIN.WITH_GENERATED_IMAGES == True:
    #         # in schedule type 2, training extra epochs with mixture of original & generated images.
    #         config.defrost()
    #         config.TRAIN.WITH_GENERATED_IMAGES = False
    #         config.freeze()
    #     for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.EXTRA_FINAL_TRAIN_EPOCH):
    #         extra_train_dataloader, _, _ = construct_dataloader(config)
    #         adjust_learning_rate(optimizer, epoch, config)
    #         # train for one epoch
    #         train_one(extra_train_dataloader, model, criterion, optimizer, epoch, config)
    #         # evaluate on validation set
    #         metric_name, metric_result, logits = validate(test_dataloader, model, criterion, epoch, config, return_logits=True)
    #         # remember best acc@1 and save checkpoint
    #         if sweep_run == False:
    #             logger.info(f"[Epoch {epoch}] Val {metric_name}: {metric_result:.3f}")
    #         # remember best acc@1 and save checkpoint
    #         if metric_result > best_acc1:
    #             model_info["best_logits"] = logits
    #         best_acc1 = max(metric_result, best_acc1)

    logger.info(f"=> Learning rate {config.TRAIN.LR}, L2 lambda {config.TRAIN.WD}: Best score: Acc@1 {best_acc1:.3f}")

    if sweep_run and config.TRAIN.SEARCH_RESULT_ON_LAST_EPOCH:
        return metric_result

    del model, criterion, optimizer
    gpu_gc()

    if sweep_run:
        return best_acc1
    else:
        return best_acc1, model_info


def IC_multi_task_train_one(train_loader, model, task_index, ds_name, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    test_metric = dataset_metrics[ds_name]
    metric = get_metric(test_metric)
    metric_name = metric.__name__

    outputs = []
    targets = []

    model.train()

    end = time.time()
    for _, batch in enumerate(train_loader):

        images, target = batch[:2]

        # measure data loading time
        data_time.update(time.time() - end)

        if len(config.GPUS) == 1:
            images = images.cuda(config.GPUS[0], non_blocking=True)

        if images.shape[0] == 1:
            continue  # TODO: check this fix on batch left is size-1
        if target.shape[-1] == 1:
            target = target[:, 0]
        target = target.cuda(config.GPUS[0], non_blocking=True)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        output = model.forward(images, task_index)

        loss = criterion(output, target)
        loss.backward()

        if config.TRAIN.CLIP_GRAD_NORM > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD_NORM)

        optimizer.step()
        losses.update(loss.item(), images.size(0))

        outputs.append(output)
        targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    logits = outputs.softmax(-1).data.cpu().numpy()
    labels = targets.data.cpu().numpy()
    # TODO: this try except block is used for addressing NaNs on metrics like mAP.
    try:
        metric_result = 100.0 * metric(labels, logits)
    except:
        metric_result = 0.0

    return metric_name, metric_result, losses.avg


@torch.no_grad()
def IC_multi_task_validate(val_loader, model, task_index, task_list, criterion, epoch, config, return_logits=False):
    batch_time = AverageMeter()
    ds_name = task_list[task_index]
    test_metric = dataset_metrics[ds_name]
    metric = get_metric(test_metric)
    metric_name = metric.__name__

    outputs = []
    targets = []

    model.eval()
    end = time.time()
    for batch in val_loader:
        images, target = batch[:2]

        if len(config.GPUS) == 1:
            images = images.cuda(config.GPUS[0], non_blocking=True)
        target = target.cuda(config.GPUS[0], non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]

        # compute output
        if config.TRAIN.MULTI_TASK_HEAD == "shared":
            output = model(images)
        elif config.TRAIN.MULTI_TASK_HEAD == "split":
            output = model(images, task_index)

        outputs.append(output)
        targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    if config.TRAIN.MULTI_TASK_HEAD == "shared":
        # caculate the cumulative class index in shared head for current task
        cls_end = sum([len(class_map[task_list[prev_index]]) for prev_index in range(task_index + 1)])
        cls_start = cls_end - len(class_map[task_list[task_index]])
        logits = outputs[:, cls_start:cls_end].softmax(-1).data.cpu().numpy()
    elif config.TRAIN.MULTI_TASK_HEAD == "split":
        logits = outputs.softmax(-1).data.cpu().numpy()
    labels = targets.data.cpu().numpy()
    # TODO: this try except block is used for addressing NaNs on metrics like mAP.
    try:
        metric_result = 100.0 * metric(labels, logits)
    except:
        metric_result = 0.0
    # logger.info(f"[Epoch {epoch}] Val: {metric_name} {metric_result:.3f}")

    if return_logits:
        return metric_name, metric_result, logits
    else:
        return metric_name, metric_result


def train_one(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    metric = get_metric(config.TEST.METRIC)
    metric_name = metric.__name__

    outputs = []
    targets = []

    model.train()

    end = time.time()
    for _, batch in enumerate(train_loader):

        images, target = batch[:2]

        # measure data loading time
        data_time.update(time.time() - end)

        if len(config.GPUS) == 1:
            images = images.cuda(config.GPUS[0], non_blocking=True)

        if images.shape[0] == 1:
            continue  # TODO: check this fix on batch left is size-1
        if target.shape[-1] == 1:
            target = target[:, 0]
        target = target.cuda(config.GPUS[0], non_blocking=True)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        output = model.forward(images)

        loss = criterion(output, target)
        loss.backward()

        if config.TRAIN.CLIP_GRAD_NORM > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD_NORM)

        optimizer.step()
        losses.update(loss.item(), images.size(0))

        outputs.append(output)
        targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    logits = outputs.softmax(-1).data.cpu().numpy()
    labels = targets.data.cpu().numpy()
    # TODO: this try except block is used for addressing NaNs on metrics like mAP.
    try:
        metric_result = 100.0 * metric(labels, logits)
    except:
        metric_result = 0.0
    # logger.info(f"[Epoch {epoch}] Train: {metric_name} {metric_result:.3f} loss {losses.avg:.3f}")
    return metric_name, metric_result, losses.avg


@torch.no_grad()
def validate(val_loader, model, criterion, epoch, config, return_logits=False):
    batch_time = AverageMeter()
    metric = get_metric(config.TEST.METRIC)
    metric_name = metric.__name__

    outputs = []
    targets = []

    model.eval()
    end = time.time()
    for batch in val_loader:
        images, target = batch[:2]

        if len(config.GPUS) == 1:
            images = images.cuda(config.GPUS[0], non_blocking=True)
        target = target.cuda(config.GPUS[0], non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]

        # compute output
        output = model(images)
        outputs.append(output)
        targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    logits = outputs.softmax(-1).data.cpu().numpy()
    labels = targets.data.cpu().numpy()
    # TODO: this try except block is used for addressing NaNs on metrics like mAP.
    try:
        metric_result = 100.0 * metric(labels, logits)
    except:
        metric_result = 0.0

    if not return_logits:
        logger.info(f"[Epoch {epoch}] Val: {metric_name} {metric_result:.3f}")

    if return_logits:
        return metric_name, metric_result, logits
    else:
        return metric_result


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.TRAIN.LR
    for milestone in config.TRAIN.SCHEDULE:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def multi_task_hyperparameter_sweep_lr(train_dataloader, val_dataloader, task_list, config):
    logger.info("=> Start hyperparameter tuning.")
    start = time.time()
    learning_rate_list = np.logspace(-6, -1, num=6).tolist()
    best_score = 0
    best_lr = 0
    best_l2_lambda = 0
    for lr_one in learning_rate_list:
        config.defrost()
        config.TRAIN.LR = lr_one
        config.freeze()
        l2_lambda, best_score_one = multi_task_hyperparameter_sweep(train_dataloader, val_dataloader, task_list, config)
        logger.info(f"=> Learning rate: {lr_one}, best_score {best_score_one}")
        if best_score < best_score_one:
            best_score = best_score_one
            best_lr = lr_one
            best_l2_lambda = l2_lambda
    logger.info(f"Hyper parameter tuning result: learning rate {best_lr}, l2_lambda {best_l2_lambda}")
    logger.info("=> Hyperparameter tuning duration time: {:.2f}s".format(time.time() - start))
    logger.info("=> Finished hyperparameter tuning.")
    return best_lr, best_l2_lambda


def hyperparameter_sweep_lr(train_dataloader, val_dataloader, config):
    logger.info("=> Start hyperparameter tuning.")
    start = time.time()
    learning_rate_list = np.logspace(-6, -1, num=6).tolist()
    best_score = 0
    best_lr = 0
    best_l2_lambda = 0
    for lr_one in learning_rate_list:
        config.defrost()
        config.TRAIN.LR = lr_one
        config.freeze()
        l2_lambda, best_score_one = hyperparameter_sweep(train_dataloader, val_dataloader, config)
        logger.info(f"=> Learning rate {lr_one}, best_score {best_score_one}")
        if best_score < best_score_one:
            best_score = best_score_one
            best_lr = lr_one
            best_l2_lambda = l2_lambda
    logger.info(f"Hyper parameter tuning result: learning rate {best_lr}, l2_lambda {best_l2_lambda}")
    logger.info("=> Hyperparameter tuning duration time: {:.2f}s".format(time.time() - start))
    logger.info("=> Finished hyperparameter tuning.")
    return best_lr, best_l2_lambda


def merge_trainval_loader(train_loader, val_loader, task_list=None):
    # TODO: DataLoader from feature.py get_dataloader()
    if type(train_loader) is list and type(val_loader) is list:
        full_set_loader_list = []
        for indx, item in enumerate(train_loader):
            trainset, valset = item.dataset, val_loader[indx].dataset
            fullset = torch.utils.data.ConcatDataset([trainset, valset])
            # assert trainset.dataset is valset.dataset
            assert len(fullset) == len(trainset) + len(valset)
            full_set_loader_list.append(
                create_dataloader(
                    fullset,
                    batch_size=train_loader[indx].batch_size,
                    shuffle=True,
                    num_workers=train_loader[indx].num_workers,
                    pin_memory=train_loader[indx].pin_memory,
                )
            )
        return full_set_loader_list
    elif type(train_loader) is torch.utils.data.DataLoader and type(val_loader) is list:
        valset_list = []
        for indx, val in enumerate(val_loader):
            valset = val.dataset
            # st = []
            # for data_item in valset.dataset.dataset.dataset_manifest.images:
            #     st.append(data_item.labels[0])
            # print(np.unique(st))
            valset_list.append(valset)
        valset_dataset = torch.utils.data.ConcatDataset(valset_list)
        trainvalset = torch.utils.data.ConcatDataset([train_loader.dataset, valset_dataset])
        return create_dataloader(
            trainvalset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
        )
    else:
        trainset, valset = train_loader.dataset, val_loader.dataset
        fullset = trainset.dataset
        assert trainset.dataset is valset.dataset
        assert len(fullset) == len(trainset) + len(valset)

        return create_dataloader(
            fullset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
        )


def full_model_finetune(train_dataloader, val_dataloader, test_dataloader, no_hyperparameter_tuning, lr, l2, config):

    if no_hyperparameter_tuning:
        best_lr = lr
        best_l2_lambda = l2
    else:
        best_lr, best_l2_lambda = hyperparameter_sweep_lr(train_dataloader, val_dataloader, config)

    logger.info("=> The final classifier is on training ...")
    logger.info(f"Hyperparameters: learning_rate = {best_lr}, l2_lambda = {best_l2_lambda}")
    config.defrost()
    config.TRAIN.LR = best_lr
    config.TRAIN.WD = best_l2_lambda
    config.TRAIN.END_EPOCH += config.TRAIN.EXTRA_FINAL_TRAIN_EPOCH
    config.freeze()

    if config.DATASET.DATASET == "patch-camelyon" and config.DATASET.NUM_SAMPLES_PER_CLASS == 10000:
        # deal with patch camelyon large dataset (search using 10000-shot subset, final run with the full dataset)
        logger.info(f"Used the subset to train the model, regenerating the full set for final run.")
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = -1
        config.freeze()
        logger.info(f"Old: len(train)={len(train_dataloader.dataset)}, len(val)={len(val_dataloader.dataset)}, len(test)={len(test_dataloader.dataset)}.")
        train_dataloader, val_dataloader, test_dataloader = construct_dataloader(config)
        logger.info(
            f"Generated: len(train)={len(train_dataloader.dataset)}, len(val)={len(val_dataloader.dataset)}, len(test)={len(test_dataloader.dataset)}."
        )

    if config.DATASET.MERGE_TRAIN_VAL_FINAL_RUN:
        trainval_dataloader = merge_trainval_loader(train_dataloader, val_dataloader)
        logger.info(f"Using the full trainval set to train final model. len(dataset)={len(trainval_dataloader.dataset)}")
    else:
        trainval_dataloader = train_dataloader
        logger.info(f"Using the train set only to train final model. len(dataset)={len(trainval_dataloader.dataset)}")
    return train_task(trainval_dataloader, test_dataloader, config)


def IC_Multitask_full_model_finetune(train_dataloader, val_dataloader, test_dataloader, task_list, no_hyperparameter_tuning, lr, l2, config):

    if no_hyperparameter_tuning:
        best_lr = lr
        best_l2_lambda = l2
    else:
        best_lr, best_l2_lambda = multi_task_hyperparameter_sweep_lr(train_dataloader, val_dataloader, task_list, config)

    logger.info("=> The final classifier is on training ...")
    logger.info(f"Hyperparameters: learning_rate = {best_lr}, l2_lambda = {best_l2_lambda}")
    config.defrost()
    config.TRAIN.LR = best_lr
    config.TRAIN.WD = best_l2_lambda
    config.TEST.METRIC = "accuracy"
    config.TRAIN.END_EPOCH += config.TRAIN.EXTRA_FINAL_TRAIN_EPOCH
    config.freeze()

    if config.DATASET.DATASET == "patch-camelyon" and config.DATASET.NUM_SAMPLES_PER_CLASS == 10000:
        # deal with patch camelyon large dataset (search using 10000-shot subset, final run with the full dataset)
        logger.info(f"Used the subset to train the model, regenerating the full set for final run.")
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = -1
        config.freeze()
        logger.info(f"Old: len(train)={len(train_dataloader.dataset)}, len(val)={len(val_dataloader.dataset)}, len(test)={len(test_dataloader.dataset)}.")
        train_dataloader, val_dataloader, test_dataloader = construct_dataloader(config)
        logger.info(
            f"Generated: len(train)={len(train_dataloader.dataset)}, len(val)={len(val_dataloader.dataset)}, len(test)={len(test_dataloader.dataset)}."
        )

    if config.DATASET.MERGE_TRAIN_VAL_FINAL_RUN:
        trainval_dataloader = merge_trainval_loader(train_dataloader, val_dataloader, task_list=task_list)
        logger.info(f"Using the full trainval set to train final model. len(dataset)={len(trainval_dataloader)}")
    else:
        trainval_dataloader = train_dataloader
        logger.info(f"Using the train set only to train final model. len(dataset)={len(trainval_dataloader)}")
    # TODO: using val loader to evaluate
    return IC_Multitask_train_task(trainval_dataloader, test_dataloader, task_list, config)
