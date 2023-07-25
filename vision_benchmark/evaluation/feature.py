"""
This module extract features with model to be evaluated and given dataset.
"""
import os
import time
from loguru import logger
import pickle
import numpy as np
import sys, json
from sklearn.model_selection import train_test_split
import timm
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torchvision.models
import torchvision.datasets
import torch.nn.functional as F
from .metric import get_metric

from ..common.constants import get_dataset_hub, VISION_DATASET_STORAGE

from ..models import *
from ..datasets import class_map, template_map

from PIL import Image
from PIL import ImageFile

from vision_datasets import ManifestDataset
from nltk.corpus import wordnet as wn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

nltk.download("punkt")
nltk.download("wordnet")

import pdb

from collections import Counter
import math
import random
import numpy as np

# The following line is to solve PIL "IOError: image file truncated" with big images.
# Refer to https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class EvalModel(nn.Module):
    def __init__(self, model_cls):
        super().__init__()
        for param in model_cls.parameters():
            param.requires_grad = False
        self.feature_model = nn.Sequential(*list(model_cls.children())[:-1])

    def forward(self, x):
        features = self.feature_model(x)
        return features


class FeatureData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=6, pin_memory=True):
    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2**32
        worker_seed = worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(0)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=None,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator if shuffle else None,
    )
    return loader


def get_dataloader(dataset, val_split=0.0, batch_size_per_gpu=64, workers=0, pin_memory=True, force_multilabel=False, total_classes=None):
    if val_split == 0:
        return create_dataloader(dataset, batch_size=batch_size_per_gpu, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    else:

        def train_val_dataset(dataset, val_split, force_multilabel=False):
            # this implementation does not generate class-balanced splits.
            # train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

            # quick fetch labels without accessing images / transformations
            def quick_fetch_labels(dataset):
                dataset_info = dataset.dataset_info
                dataset_manifest = dataset.dataset.dataset_manifest
                from vision_datasets import DatasetTypes

                if force_multilabel is True:
                    labels = [multilabel_to_vec(x.labels, total_classes) for x in dataset_manifest.images]
                elif dataset_info.type == DatasetTypes.IC_MULTILABEL:
                    labels = [multilabel_to_vec(x.labels, len(dataset.labels)) for x in dataset_manifest.images]
                elif dataset_info.type == DatasetTypes.IC_MULTICLASS:
                    labels = [multiclass_to_int(x.labels) for x in dataset_manifest.images]
                else:
                    raise NotImplementedError
                return np.asarray(labels)

            logger.debug("Quick fetch label starts.")
            labels = quick_fetch_labels(dataset)
            logger.debug("Quick fetch label finished.")
            # logger.debug('Full fetch label starts.')
            # labels_all_fetch = np.asarray([x[1] for x in dataset])
            # logger.debug('Full fetch label finished.')
            # assert (labels == labels_all_fetch).all()
            # logger.debug('Quick fetch label same as full fetch.')

            # FIX: class-balanced split generation
            if len(labels.shape) == 1:
                # single-class IC datasets
                cls_to_count = Counter(labels)
                val_indices = []

                for label in cls_to_count:
                    n_samples = math.ceil(cls_to_count[label] * val_split)
                    samples = np.where(labels == label)[0][:n_samples]  # TODO: not doing random. confirm that it is unnecessary
                    val_indices.append(samples)
                val_idx = set(np.concatenate(val_indices).tolist())
                train_idx = set(list(range(len(dataset)))) - val_idx
                train_idx, val_idx = list(train_idx), list(val_idx)
            elif len(labels.shape) == 2:
                # multi-class IC datasets
                val_target_count = np.ceil(np.sum(labels, axis=0) * val_split)
                next_targets = np.where(val_target_count > 0)[0]
                val_idx = []

                while next_targets.size > 0:
                    target_cls = next_targets[0]
                    next_sample = np.where(labels[:, target_cls] > 0)[0][0]
                    val_idx.append(next_sample)
                    val_target_count -= labels[next_sample]
                    labels[next_sample] = 0
                    next_targets = np.where(val_target_count > 0)[0]

                val_idx = np.asarray(val_idx).tolist()
                train_idx = set(list(range(len(dataset)))) - set(val_idx)
                train_idx = list(train_idx)
            else:
                raise NotImplementedError

            # val_idx, train_idx = np.split(list(range(len(dataset))), [int(len(dataset)*val_split)])
            # train_idx, val_idx = [x.tolist() for x in (train_idx, val_idx)]
            return {"train": Subset(dataset, train_idx), "val": Subset(dataset, val_idx)}

        datasets = train_val_dataset(dataset, val_split, force_multilabel=force_multilabel)
        train_loader = create_dataloader(datasets["train"], batch_size=batch_size_per_gpu, shuffle=True, num_workers=workers, pin_memory=pin_memory)
        val_loader = create_dataloader(datasets["val"], batch_size=batch_size_per_gpu, shuffle=False, num_workers=workers, pin_memory=pin_memory)
        return train_loader, val_loader


def load_custom_ic_model(config):
    logger.debug(f"=> Loading custom model {config.MODEL.NAME}.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device("cuda")
    logger.debug(f"Load model in device {device}.")

    model = eval(config.MODEL.NAME + ".get_cls_model")(config)
    model_file = config.TEST.MODEL_FILE
    logger.debug(f"=> load model file: {model_file}")
    ext = model_file.split(".")[-1]
    if ext == "pth":
        state_dict = torch.load(model_file, map_location="cpu")
    elif ext == "pkl":
        logger.debug("=> load pkl model")
        with open(model_file, "rb") as f:
            state_dict = pickle.load(f)["model"]

        for k, v in state_dict.items():
            state_dict[k] = torch.from_numpy(v)
    else:
        raise ValueError(f"=> Unknown model file, with ext {ext}")
    model.load_state_dict(state_dict)
    return model


def load_custom_zeroshot_model(config):
    logger.debug(f"=> Loading custom model {config.MODEL.NAME}.")
    torch.device("cuda")

    model = eval(config.MODEL.NAME + ".get_zeroshot_model")(config)
    model_file = config.TEST.MODEL_FILE
    logger.debug(f"=> load model file: {model_file}")
    ext = model_file.split(".")[-1]
    if ext == "pth" or ext == "pt":
        state_dict = torch.load(model_file, map_location="cpu")
    elif ext == "pkl":
        logger.debug("=> load pkl model")
        with open(model_file, "rb") as f:
            state_dict = pickle.load(f)["model"]

        for k, v in state_dict.items():
            state_dict[k] = torch.from_numpy(v)
    else:
        raise ValueError(f"=> Unknown model file, with ext {ext}")
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"loading checkpoint msg: {msg}")
    return model


def get_model(config, feature_type="image"):
    model_name = config.MODEL.NAME
    if model_name in dir(torchvision.models):
        model_pretrained = eval("torchvision.models." + model_name)(pretrained=True)
        model = EvalModel(model_pretrained)
        logger.debug(f"Using Pytorch pretrained model {model_name}")
    elif model_name in timm.list_models(pretrained=True):
        model = timm.create_model(model_name, pretrained=True)
        if model_name.startswith("efficientnet"):
            model = EvalModel(model)
        elif model_name.startswith("vit") or model_name.startswith("deit"):
            model.forward = model.forward_features
        else:
            raise Exception("Please define Timm feature extraction model.")
        logger.debug(f"Using Timm pretrained model {model_name}")
    elif model_name.startswith("cls_"):
        model = load_custom_ic_model(config)
        model.forward = model.forward_features
    elif model_name.startswith("mae_"):
        model = mae.get_model(config)
        model.forward = model.forward_features
    elif model_name.startswith("declip_") or model_name.startswith("slip_") or model_name.startswith("clip_yfcc_"):
        model = declip.get_model(config)
        if feature_type == "image":
            model.forward = model.encode_image
        elif feature_type == "text":
            model.forward = model.encode_text
        else:
            raise Exception("Incorrect model type.")
        if not config.MODEL.CLIP_FP32:
            model.half()
    elif model_name.startswith("filip_") or model_name.startswith("defilip_"):
        model = declip.get_model(config)
        if feature_type == "image":
            model.forward = model.encode_image_dense
        elif feature_type == "text":
            model.forward = model.encode_text_dense
        else:
            raise Exception("Incorrect model type.")
        if not config.MODEL.CLIP_FP32:
            model.half()
    elif model_name.startswith("mocov3_"):
        model = mocov3.get_model(config)
        model.forward = model.forward_features
    elif model_name.startswith("clip_"):
        model = load_custom_zeroshot_model(config)

        if config.LOSS.LOSS == "softmax":
            if feature_type == "image":
                model.forward = model.encode_image
            elif feature_type == "text":
                model.forward = model.encode_text
            else:
                raise Exception("Incorrect model type")
        elif config.LOSS.LOSS == "contrast":
            logger.debug(f"Training objective: { config.LOSS.LOSS }.")

    else:
        if config.MODEL.CLIP_FP32:
            import clip_vlp as clip
        else:
            import clip
        if model_name in clip.available_models():
            model, _ = clip.load(model_name, jit=False)
            if feature_type == "image":
                model.forward = model.encode_image
            elif feature_type == "text":
                model.forward = model.encode_text
            else:
                raise Exception("Incorrect model type.")
            logger.debug(f"Using CLIP pretrained model {model_name}, input size {model.visual.input_resolution}")
        else:
            raise ValueError(f"Unknown model name {model_name}.")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.debug(f"Number of params: {pytorch_total_params / 1000000}M.")
    return model


def extract_feature(model, data_loader, config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    # Generate model statistics
    visual_backbone = model.visual if model.visual is not None else model
    model_info = config.MODEL.STATS
    config.defrost()
    model_info["n_visual_params"] = sum(p.numel() for p in visual_backbone.parameters())
    model_info["n_backbone_params"] = sum(p.numel() for p in model.parameters())
    model_info["n_params"] = sum(p.numel() for p in model.parameters())
    config.freeze()

    start = time.time()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, f"Extracting features with model {config.MODEL.NAME}.", disable=config.TEST.DISABLE_TQDM):
            x, y = batch[:2]
            # compute output
            if device == torch.device("cuda"):
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
            outputs = model(x)
            all_features.append(outputs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    logger.debug(f"=> Feature extraction duration time: {time.time() - start:.2f}s")
    return np.reshape(features, (features.shape[0], -1)), np.reshape(labels, (labels.shape[0], -1))


def multilabel_to_vec(indices, n_classes):
    vec = np.zeros(n_classes)
    for x in indices:
        vec[x] = 1
    return vec


def multiclass_to_int(indices):
    return indices[0]


def extract_features(config, feature_type="image", test_split_only=False):
    model = get_model(config, feature_type=feature_type)

    train_dataloader, val_dataloader, test_dataloader = construct_dataloader(config, feature_type="image", test_split_only=False)

    test_features, test_labels = extract_feature(model, test_dataloader, config)
    if test_split_only:
        return test_features, test_labels
    train_features, train_labels = extract_feature(model, train_dataloader, config)
    val_features, val_labels = extract_feature(model, val_dataloader, config)
    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def hypernyms_chain(concept):
    ss = wn.synsets(concept)
    hypernyms_chain = []
    # chain_list = ss.hypernym_paths()

    while len(ss) > 0:
        ss = ss[0]

        hypernyms_chain.append(ss.lemmas()[0].name())
        # print(f'{ss.name()}, {ss.definition()}, {ss.hypernyms()}')
        ss = ss.hypernyms()

    hypernyms_chain = " ".join(hypernyms_chain)
    return hypernyms_chain


def concept_definition(concept):
    ss = wn.synsets(concept)
    if len(ss) > 0:
        definition = ss[0].definition()
    else:
        definition = ""

    return definition


@torch.no_grad()
# TOOD: replace flag: generated
def extract_text_features(config, dataset_name, tokenizer, args=None, model=None, return_numpy=True):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    class_names = class_map.get(dataset_name.replace("-generated", ""))
    if not class_names:
        hub = get_dataset_hub()
        from vision_datasets import Usages

        manifest = hub.create_dataset_manifest(VISION_DATASET_STORAGE, local_dir=config.DATASET.ROOT, name=dataset_name, usage=Usages.TEST_PURPOSE)
        if manifest:
            class_names = manifest[0].labelmap

    if config.KNOWLEDGE.WIKITIONARY.USE_DEFINITION:
        wiki_path = config.KNOWLEDGE.WIKITIONARY.WIKI_DICT_PATH
        wiki_tsv_path = os.path.join(wiki_path, dataset_name.replace("-generated", "") + "_knowledge.tsv")
        wiki_anwser_list = json.load(open(wiki_tsv_path, encoding="utf-8"))

        count_has_wiki_knowledge = 0
        wiki_dict = {}
        for k2v in wiki_anwser_list:
            wiki_dict[k2v["classname"]] = k2v["def_wiki"]
            if k2v["def_wiki"]:
                count_has_wiki_knowledge += 1
        logger.debug(f"coverage is {count_has_wiki_knowledge} / {len(wiki_dict)}")

    if config.KNOWLEDGE.WORDNET.USE_DEFINITION:
        wiki_path = config.KNOWLEDGE.WIKITIONARY.WIKI_DICT_PATH
        wiki_tsv_path = os.path.join(wiki_path, dataset_name.replace("-generated", "") + "_knowledge.tsv")
        wiki_anwser_list = json.load(open(wiki_tsv_path, encoding="utf-8"))

        count_has_wiki_knowledge = 0
        wiki_dict = {}
        for k2v in wiki_anwser_list:
            wiki_dict[k2v["classname"]] = k2v["def_wn"]
            if k2v["def_wn"]:
                count_has_wiki_knowledge += 1
        logger.debug(f"coverage is {count_has_wiki_knowledge} / {len(wiki_dict)}")

    if config.KNOWLEDGE.WORDNET.USE_HIERARCHY:
        wiki_path = config.KNOWLEDGE.WIKITIONARY.WIKI_DICT_PATH
        wiki_tsv_path = os.path.join(wiki_path, dataset_name.replace("-generated", "") + "_knowledge.tsv")
        wiki_anwser_list = json.load(open(wiki_tsv_path, encoding="utf-8"))

        count_has_wiki_knowledge = 0
        wiki_dict = {}
        for k2v in wiki_anwser_list:
            if len(k2v["path_wn"]) > 0:
                path_length = min(3, len(k2v["path_wn"]))
                path_wn = " ".join(k2v["path_wn"][:path_length])

            else:
                path_wn = k2v["path_wn"]
            wiki_dict[k2v["classname"]] = path_wn
            if k2v["path_wn"]:
                count_has_wiki_knowledge += 1
        logger.debug(f"coverage is {count_has_wiki_knowledge} / {len(wiki_dict)}")

    if config.KNOWLEDGE.GPT3.USE_GPT3:
        gpt3_path = config.KNOWLEDGE.GPT3.GPT3_DICT_PATH
        gpt3_tsv_path = os.path.join(gpt3_path, "GPT3_" + dataset_name.replace("-generated", "") + ".tsv")
        gpt3_anwser_list = json.load(open(gpt3_tsv_path, encoding="utf-8"))

        gpt3_dict = {}
        for k2v in gpt3_anwser_list:
            gpt3_dict[k2v["classname"]] = k2v["gpt3"]

    if args is not None and args.text_feature_only:
        return wiki_dict, gpt3_dict

    templates = template_map.get(dataset_name.replace("-generated", ""), ["a photo of a {}"])
    if model is None:
        model = get_model(config, feature_type="text")
    start = time.time()
    model.to(device)
    model.eval()

    zeroshot_weights = []
    wiki_count, gpt3_count = 0, 0
    for classname in tqdm(class_names, f"Extracting text features with model {config.MODEL.NAME}.", disable=config.TEST.DISABLE_TQDM):
        if type(classname) == list:
            classname = classname[0]

        knowledge_text_list = []
        if config.KNOWLEDGE.WIKITIONARY.USE_DEFINITION or config.KNOWLEDGE.WORDNET.USE_DEFINITION or config.KNOWLEDGE.WORDNET.USE_HIERARCHY:
            if classname in wiki_dict:
                knowledge_text_list.append(wiki_dict[classname])
                wiki_count += 1

        if config.KNOWLEDGE.GPT3.USE_GPT3:
            if config.KNOWLEDGE.AGGREGATION.MEHTOD == "WIKI_AND_GPT3":
                for knowledge_text in gpt3_dict[classname][: config.KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS]:
                    knowledge_text_list.append(knowledge_text)
                    gpt3_count += 1

            elif config.KNOWLEDGE.AGGREGATION.MEHTOD == "WIKI_THEN_GPT3" and len(knowledge_text_list) == 0:
                for knowledge_text in gpt3_dict[classname][: config.KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS]:
                    knowledge_text_list.append(knowledge_text)
                    gpt3_count += 1

        knowledge_text_list_aug = []
        for knowledge_text in knowledge_text_list:
            knowledge_text = f" ; {classname} , " + knowledge_text if knowledge_text is not None else ""
            knowledge_text = " " + " ".join(word_tokenize(knowledge_text))
            knowledge_text_list_aug.append(knowledge_text)

        if len(knowledge_text_list_aug) == 0:
            texts = [template.format(classname) for template in templates]
        else:
            texts = [template.format(classname) + knowledge_text for knowledge_text in knowledge_text_list_aug for template in templates]

        if not config.MODEL.SPEC.TEXT.get("SKIP_TOKENIZE", False):
            texts = tokenizer(texts, context_length=config.MODEL.SPEC.TEXT.CONTEXT_LENGTH).to(device)

        if config.MODEL.SPEC.get("DENSE_EVAL", False):
            class_embeddings = model.encode_text_dense(texts)
        else:
            class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    logger.debug(f"=> Feature extraction duration time: {time.time() - start:.2f}s")
    logger.debug(f"=> Knowledge source count | knowledge_count: {wiki_count} | gpt3_count {gpt3_count} ")

    if return_numpy:
        return zeroshot_weights.cpu().detach().numpy()
    else:
        return zeroshot_weights

from torchvision.transforms import InterpolationMode
def construct_dataloader(config, feature_type="image", test_split_only=False):
    if config.DATASET.CENTER_CROP:
        logger.debug("Do center crop")
        transform_clip = transforms.Compose(
            [
                transforms.Resize(config.TRAIN.IMAGE_SIZE[0], interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=config.TRAIN.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.INPUT.MEAN, std=config.INPUT.STD),
            ]
        )
    else:
        logger.debug("no center crop")
        transform_clip = transforms.Compose(
            [
                transforms.Resize(config.TRAIN.IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.INPUT.MEAN, std=config.INPUT.STD),
            ]
        )

    from vision_datasets import Usages, DatasetTypes
    from vision_datasets.pytorch import TorchDataset

    hub = get_dataset_hub()
    dataset_names = set([x["name"] for x in hub.list_data_version_and_types()])
    if config.DATASET.DATASET in dataset_names:
        vision_dataset_storage = "https://cvinthewildeus.blob.core.windows.net/datasets"
        local_temp = config.DATASET.ROOT

        # return [manifest, dataset_info, downloader_resources]
        results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, config.DATASET.DATASET, usage=Usages.TEST_PURPOSE)
        if results:
            test_set, test_set_dataset_info, _ = results
        logger.info(f"Test size is {len(test_set.images)}.")

        # re-define transform_clip to organize the labels
        if test_set_dataset_info.type == DatasetTypes.IC_MULTILABEL:
            previous_transform = transform_clip

            def transform_clip(x, y):
                test_set_ = ManifestDataset(test_set_dataset_info, test_set)
                return (previous_transform(x), multilabel_to_vec(y, len(test_set_.labels)))

        elif test_set_dataset_info.type == DatasetTypes.IC_MULTICLASS:
            previous_transform = transform_clip

            def transform_clip(x, y):
                return (previous_transform(x), multiclass_to_int(y))

        test_dataloader = get_dataloader(TorchDataset(ManifestDataset(test_set_dataset_info, test_set), transform=transform_clip))
        # download train/val split only if test_split_only is False
        if not test_split_only:
            train_set_results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, config.DATASET.DATASET, usage=Usages.TRAIN_PURPOSE)
            if train_set_results:
                train_set, train_set_dataset_info, _ = train_set_results

            logger.info(f"Loading images from {train_set_dataset_info.root_folder}/{train_set_dataset_info.train_path}...")

            if config.TRAIN.WITH_GENERATED_IMAGES:
                train_set_generated_results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, f"{config.DATASET.DATASET}-generated", usage=Usages.TRAIN_PURPOSE)
                if train_set_generated_results:
                    train_set_generated, train_set_generated_dataset_info, _ = train_set_generated_results
            
            if config.TRAIN.WITH_RETRIEVAL_IMAGES:
                train_set_retrieval_results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, f"{config.DATASET.DATASET}-retrieval", usage=Usages.TRAIN_PURPOSE)
                if train_set_retrieval_results:
                    train_set_retrieval, train_set_retrieval_dataset_info, _ = train_set_retrieval_results
                    
            val_set = None
            val_set_results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, config.DATASET.DATASET, usage=Usages.VAL_PURPOSE)
            if val_set_results:
                val_set, val_set_dataset_info, _ = val_set_results

            # few-shot dataset construction
            if config.DATASET.NUM_SAMPLES_PER_CLASS > 0:
                num_samples_per_class = config.DATASET.NUM_SAMPLES_PER_CLASS
                random_seed = config.DATASET.RANDOM_SEED_SAMPLING
                train_set = train_set.sample_few_shot_subset(num_samples_per_class, random_seed)

            if config.TRAIN.WITH_GENERATED_IMAGES:
                logger.info(f"Train set size: {len(train_set.images)}")
                if config.TRAIN.WGEN_PER_CLASS > 0:
                    train_set_generated = train_set_generated.sample_few_shot_subset(config.TRAIN.WGEN_PER_CLASS, config.DATASET.RANDOM_SEED_SAMPLING)
                logger.info(f"Generated set size: {len(train_set_generated.images)}")
                train_set.images += train_set_generated.images
                logger.info(f"Train set size after adding generated images: {len(train_set.images)}")

            if config.TRAIN.WITH_RETRIEVAL_IMAGES:
                logger.info(f"Train set size: {len(train_set.images)}")
                if config.TRAIN.WRET_PER_CLASS > 0:
                    train_set_retrieval = train_set_retrieval.sample_few_shot_subset(config.TRAIN.WRET_PER_CLASS, config.DATASET.RANDOM_SEED_SAMPLING)
                logger.info(f"Retrieval set size: {len(train_set_retrieval.images)}")
                train_set.images += train_set_retrieval.images
                logger.info(f"Train set size after adding retrieval images: {len(train_set.images)}")

            val_split = 0.2
            train_dataloader, val_dataloader = get_dataloader(TorchDataset(ManifestDataset(train_set_dataset_info, train_set), transform=transform_clip), val_split=val_split)
            logger.info(f"Val split from Train set: Train size is {len(train_set.images)*(1-val_split)}, and validation size is {len(train_set.images)*val_split}.")
    else:
        if not test_split_only:
            if config.DATASET.VAL_SET:
                train_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET), transform=transform_clip))
                val_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.VAL_SET), transform=transform_clip))
            else:
                train_dataloader, val_dataloader = get_dataloader(
                    torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET), transform=transform_clip), val_split=0.2
                )
        test_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET), transform=transform_clip))

    return train_dataloader, val_dataloader, test_dataloader


import copy


def split_train_val_dataset(dataset, val_split, force_multilabel=False, total_classes=None):
    # this implementation does not generate class-balanced splits.
    # train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

    # quick fetch labels without accessing images / transformations
    def quick_fetch_labels(dataset):
        dataset_info = dataset.dataset_info
        dataset_manifest = dataset.dataset.dataset_manifest
        from vision_datasets import DatasetTypes

        if force_multilabel is True:
            labels = [multilabel_to_vec(x.labels, n_classes=total_classes) for x in dataset_manifest.images]
        elif dataset_info.type == DatasetTypes.IC_MULTILABEL:
            labels = [multilabel_to_vec(x.labels, len(dataset.labels)) for x in dataset_manifest.images]
        elif dataset_info.type == DatasetTypes.IC_MULTICLASS:
            labels = [multiclass_to_int(x.labels) for x in dataset_manifest.images]
        else:
            raise NotImplementedError
        return np.asarray(labels)

    logger.debug("Quick fetch label starts.")
    labels = quick_fetch_labels(dataset)
    logger.debug("Quick fetch label finished.")
    # logger.debug('Full fetch label starts.')
    # labels_all_fetch = np.asarray([x[1] for x in dataset])
    # logger.debug('Full fetch label finished.')
    # assert (labels == labels_all_fetch).all()
    # logger.debug('Quick fetch label same as full fetch.')

    # FIX: class-balanced split generation
    if len(labels.shape) == 1:
        # single-class IC datasets
        cls_to_count = Counter(labels)
        val_indices = []

        for label in cls_to_count:
            n_samples = math.ceil(cls_to_count[label] * val_split)
            samples = np.where(labels == label)[0][:n_samples]  # TODO: not doing random. confirm that it is unnecessary
            val_indices.append(samples)
        val_idx = set(np.concatenate(val_indices).tolist())
        train_idx = set(list(range(len(dataset)))) - val_idx
        train_idx, val_idx = list(train_idx), list(val_idx)
    elif len(labels.shape) == 2:
        # multi-class IC datasets
        val_target_count = np.ceil(np.sum(labels, axis=0) * val_split)
        next_targets = np.where(val_target_count > 0)[0]
        val_idx = []

        while next_targets.size > 0:
            target_cls = next_targets[0]
            next_sample = np.where(labels[:, target_cls] > 0)[0][0]
            val_idx.append(next_sample)
            val_target_count -= labels[next_sample]
            labels[next_sample] = 0
            next_targets = np.where(val_target_count > 0)[0]

        val_idx = np.asarray(val_idx).tolist()
        train_idx = set(list(range(len(dataset)))) - set(val_idx)
        train_idx = list(train_idx)
    else:
        raise NotImplementedError

    # val_idx, train_idx = np.split(list(range(len(dataset))), [int(len(dataset)*val_split)])
    # train_idx, val_idx = [x.tolist() for x in (train_idx, val_idx)]
    return {"train": Subset(dataset, train_idx), "val": Subset(dataset, val_idx)}


# construct multi task dataloader for IC in the wild task
def construct_IC_multitask_dataloader(config, feature_type="image", test_split_only=False):
    if config.DATASET.CENTER_CROP:
        logger.debug("Do center crop")
        transform_clip = transforms.Compose(
            [
                transforms.Resize(config.TRAIN.IMAGE_SIZE[0], interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=config.TRAIN.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.INPUT.MEAN, std=config.INPUT.STD),
            ]
        )
    else:
        logger.debug("no center crop")
        transform_clip = transforms.Compose(
            [
                transforms.Resize(config.TRAIN.IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.INPUT.MEAN, std=config.INPUT.STD),
            ]
        )

    from vision_datasets import Usages, DatasetTypes
    from vision_datasets.pytorch import TorchDataset

    hub = get_dataset_hub()
    dataset_names = set([x["name"] for x in hub.list_data_version_and_types()])
    # remove finetune on imagenet-1k
    dataset_names.remove("imagenet-1k")

    vision_dataset_storage = "https://cvinthewildeus.blob.core.windows.net/datasets"
    local_temp = config.DATASET.ROOT

    test_dataloader_list = []
    train_dataloader_list = []
    val_dataloader_list = []

    test_dataset_list = []
    train_dataset_list = []
    val_dataset_list = []

    previous_transform = transform_clip

    # to make sure the order of datasets the same for each run
    # if config.TRAIN.MULTI_TASK_HEAD == "shared":
    #     dataset_names.remove("voc-2007-classification")

    dataset_names = sorted(list(dataset_names))
    total_classes = sum([len(class_map[x]) for x in dataset_names])
    # to arrange voc's multi-label index, put this dataset first.
    # dataset_names = ["voc-2007-classification"] + dataset_names
    def transform_shared_head_multi_label(x, y):
        return (previous_transform(x), multilabel_to_vec(y, total_classes))

    def transform_clip_multi_class(x, y):
        return (previous_transform(x), multiclass_to_int(y))

    task_list = []
    # for shared head, each class should be added to a base index.
    # 0-102, 103-205, 206-308, 309-411, 412-514, 515-617, 618-720, 721-823, 824-926, 927-1029
    if os.path.exists("./zeroshot_weights/embed_dist.pt"):
        embedding_sim = torch.load("./zeroshot_weights/embed_dist.pt").cpu().numpy()
    else:
        embedding_sim = None

    cls_index_base = 0
    avg_labels = []
    avg_labels_extended = []
    for name in dataset_names:
        results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, name, usage=Usages.TEST_PURPOSE)
        if results:
            test_set, test_set_dataset_info, _ = results

        if test_set_dataset_info.type == DatasetTypes.IC_MULTILABEL:
            logger.debug(f"Construct dataloader for {name} with type of {DatasetTypes.IC_MULTILABEL}.")

            def transform_clip_multi_label(x, y):
                test_set_ = ManifestDataset(test_set_dataset_info, test_set)
                return (previous_transform(x), multilabel_to_vec(y, len(test_set_.labels)))

            transform_clip = transform_clip_multi_label
        elif test_set_dataset_info.type == DatasetTypes.IC_MULTICLASS:
            logger.debug(f"Construct dataloader for {name} with type of {DatasetTypes.IC_MULTICLASS}.")
            transform_clip = transform_clip_multi_class
        else:
            logger.debug("Dataset type not supported.")

        logger.debug(f"Dataset {name}: Test size is {len(test_set.images)}.")
        test_dataset = TorchDataset(ManifestDataset(test_set_dataset_info, test_set), transform=transform_clip)

        # if config.TRAIN.MULTI_TASK_HEAD == "shared":
        #     for item in test_dataset.dataset.dataset_manifest.images:
        #         item.labels[0] = item.labels[0] + cls_index_base

        test_loader = get_dataloader(test_dataset)
        test_dataset_list.append(test_dataset)
        test_dataloader_list.append(test_loader)

        # download train/val split only if test_split_only is False
        if not test_split_only:
            train_set_results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, name, usage=Usages.TRAIN_PURPOSE)
            if train_set_results:
                train_set, train_set_dataset_info, _ = train_set_results

            val_set = None
            val_set_results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, name, usage=Usages.VAL_PURPOSE)
            if val_set_results:
                val_set, val_set_dataset_info, _ = val_set_results

            # few-shot dataset construction
            if config.DATASET.NUM_SAMPLES_PER_CLASS > 0:
                num_samples_per_class = config.DATASET.NUM_SAMPLES_PER_CLASS
                random_seed = config.DATASET.RANDOM_SEED_SAMPLING
                train_set = train_set.sample_few_shot_subset(num_samples_per_class, random_seed)

        val_split = 0.2
        if config.TRAIN.MULTI_TASK_HEAD == "shared":
            train_val_dataset = TorchDataset(ManifestDataset(train_set_dataset_info, train_set), transform=transform_shared_head_multi_label)
            train_val_dataset = split_train_val_dataset(train_val_dataset, val_split, force_multilabel=True, total_classes=total_classes)
        else:
            train_val_dataset = TorchDataset(ManifestDataset(train_set_dataset_info, train_set), transform=transform_clip)
            train_val_dataset = split_train_val_dataset(train_val_dataset, val_split)

        train_dataset = train_val_dataset["train"]
        val_dataset = copy.deepcopy(train_val_dataset["val"])
        val_dataset.dataset.transform = transform_clip

        if config.TRAIN.MULTI_TASK_HEAD == "shared":
            for item in train_dataset.dataset.dataset.dataset_manifest.images:
                for idx in range(len(item.labels)):
                    item.labels[idx] = item.labels[idx] + cls_index_base

            cls_index_base_end = cls_index_base + len(train_dataset.dataset.dataset.labels)
            cur_avg_label_per_task = sum([len(item.labels) for item in train_dataset.dataset.dataset.dataset_manifest.images]) / len(
                train_dataset.dataset.dataset.dataset_manifest.images
            )
            avg_labels.append(cur_avg_label_per_task)

            # find similar classes across tasks for each class
            threshold = config.TRAIN.SIMILARITY_THRESHOLD
            if embedding_sim is not None:
                for item in train_dataset.dataset.dataset.dataset_manifest.images:
                    across_task_index = []
                    for idx in range(len(item.labels)):
                        emb_sim_row = embedding_sim[item.labels[idx]]
                        select_index = np.where(emb_sim_row > threshold)[0]
                        across_task_index.extend([x for x in select_index if x < cls_index_base or x > cls_index_base_end])
                        # print(across_task_index)
                    if len(across_task_index) > 0:
                        item.labels.extend(across_task_index)

            cur_avg_label_per_task_extended = sum([len(item.labels) for item in train_dataset.dataset.dataset.dataset_manifest.images]) / len(
                train_dataset.dataset.dataset.dataset_manifest.images
            )
            avg_labels_extended.append(cur_avg_label_per_task_extended)

        # st = []
        # for data_item in train_dataset.dataset.dataset.dataset_manifest.images:
        #     st.append(data_item.labels[0])
        # print(np.unique(st))

        # st = []
        # for data_item in val_dataset.dataset.dataset.dataset_manifest.images:
        #     st.append(data_item.labels[0])
        # print(np.unique(st))

        # update the base index for next dataset
        cls_index_base += len(class_map[name])
        train_dataloader = get_dataloader(train_dataset)
        val_dataloader = get_dataloader(val_dataset)
        # train_dataset = TorchDataset(ManifestDataset(train_set_dataset_info, train_set), transform=transform_clip)
        # train_loader = get_dataloader(train_dataset)
        train_dataloader_list.append(train_dataloader)
        # print(next(iter(train_dataloader))[1].shape)
        val_dataloader_list.append(val_dataloader)
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)
        task_list.append(name)

        # val_split=0.2
        # train_dataloader, val_dataloader = get_dataloader(TorchDataset( ManifestDataset(train_set_dataset_info, train_set), transform=transform_clip), val_split=val_split)
        logger.debug(f"Val split from Train set: Train size is {len(train_set.images)*(1-val_split)}, and validation size is {len(train_set.images)*val_split}.")

    for idx, item in enumerate(avg_labels):
        logger.debug(f"Average labels for task {task_list[idx]}, before extending: {item:.3f}, after extending: {avg_labels_extended[idx]:.3f}")

    assert cls_index_base == sum([len(class_map[name]) for name in dataset_names])
    if config.TRAIN.MULTI_TASK_HEAD == "split":
        return task_list, train_dataloader_list, val_dataloader_list, test_dataloader_list
    elif config.TRAIN.MULTI_TASK_HEAD == "shared":
        all_train_dataloader = get_dataloader(torch.utils.data.ConcatDataset(train_dataset_list), force_multilabel=True, total_classes=total_classes)
        # all_val_dataloader = get_dataloader(torch.utils.data.ConcatDataset(val_dataset_list))
        # all_test_dataloader = get_dataloader(torch.utils.data.ConcatDataset(test_dataset_list))
        return task_list, all_train_dataloader, val_dataloader_list, test_dataloader_list
