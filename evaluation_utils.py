import json
import os
import subprocess
import zipfile

import clip
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from vision_benchmark.config import config, update_config
from vision_benchmark.datasets import HFPTTokenizer, SimpleTokenizer
from vision_benchmark.evaluation import extract_features, extract_text_features
from vision_benchmark.evaluation.metric import get_metric

import wandb
# from torchmetrics.image import FID
from tools.fid import FID

def run_elevator(elevator_root_path, args):
    shot = args.clip_selection
    if args.dataset_type is not None:
        full_dataset_name = f"{args.dataset_name}_{args.dataset_type}_{args.wandb_tag}_{args.clip_selection}"
    else:
        full_dataset_name = f"{args.dataset_name}"
    # run Elevator training on generated dataset
    cmd = f"pwd; python -m vision_benchmark.commands.linear_probe --ds=./vision_benchmark/resources/datasets/{full_dataset_name}.yaml \
        --model=./vision_benchmark/resources/model/vitb32_CLIP.yaml --no-tuning={args.no_tuning} --lr=1e-4 --l2=1e-6 \
        MODEL.CLIP_FP32 True DATASET.NUM_SAMPLES_PER_CLASS {shot} DATASET.ROOT {elevator_root_path}/datasets DATASET.DATASET {full_dataset_name} \
        OUTPUT_DIR {elevator_root_path}/log DATASET.RANDOM_SEED_SAMPLING 0 DATASET.MERGE_TRAIN_VAL_FINAL_RUN True \
        TRAIN.FREEZE_IMAGE_BACKBONE True TRAIN.MULTI_TASK False TRAIN.INIT_HEAD_WITH_TEXT_ENCODER True TRAIN.MERGE_ENCODER_AND_HEAD_PROJ False \
        KNOWLEDGE.WORDNET.USE_HIERARCHY False KNOWLEDGE.WORDNET.USE_DEFINITION False KNOWLEDGE.WIKITIONARY.USE_DEFINITION False \
        KNOWLEDGE.GPT3.USE_GPT3 False KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS 0 TRAIN.WITH_GENERATED_IMAGES {args.with_generated} TRAIN.WITH_RETRIEVAL_IMAGES {args.with_retrieval} WANDB_TAG {args.wandb_tag}"
    logger.info(f"Running Elevator training on generated dataset: {cmd}")
    subp = subprocess.Popen(cmd, shell=True)
    subp.wait()


def load_or_extract_features(args, cfg, test_split_only=False):
    if cfg.MODEL.SPEC.TEXT.TOKENIZER == "clip":
        tokenizer = SimpleTokenizer()
    elif "hf_" in cfg.MODEL.SPEC.TEXT.TOKENIZER:
        tokenizer = HFPTTokenizer(pt_name=cfg.MODEL.SPEC.TEXT.TOKENIZER[3:])
    else:
        tokenizer = None

    all_features_and_labels = extract_features(cfg, test_split_only=test_split_only)
    text_features = extract_text_features(cfg, cfg.DATASET.DATASET, tokenizer, args)

    return all_features_and_labels, text_features

@torch.no_grad()
def zeroshot_metric(args):
    args.cfg = args.ds
    update_config(config, args)
    args.cfg = args.model
    update_config(config, args)
    config.defrost()
    config.NAME = ""
    config.freeze()
    metric = get_metric(config.TEST.METRIC)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_features_and_labels, text_features = load_or_extract_features(args, config)
    print("Executing zeroshot retrieval evaluation...")
    (
        original_train_features,
        original_train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = all_features_and_labels

    if args.with_ext_train != "original":
        ext_train_zipfile_path = os.path.join(args.output_dir, f"{args.wandb_tag}_{args.clip_selection}_train.zip")
        ext_train_zipfile = zipfile.ZipFile(ext_train_zipfile_path, "r")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        feat_list = []
        label_list = []

        if args.dataset_name in ["resisc45_clip", "kitti-distance", "hateful-memes"]:
            train_metafile_path = os.path.join(args.output_dir, f"{args.wandb_tag}_{args.clip_selection}_train.json")
            with open(train_metafile_path, "r") as fread:
                train_metafile = json.load(fread)

            annos = train_metafile["annotations"]
            for item in train_metafile["images"]:
                image_id = item["id"]
                image_path = item["file_name"].split("@")[1]
                # TODO: seems json file dataset is not in order from 0?
                anno = annos[image_id - 1]
                assert anno["image_id"] == image_id
                image = ext_train_zipfile.open(image_path)
                image = preprocess(Image.open(image)).unsqueeze(0).to(device)
                image_features = clip_model.encode_image(image)
                feat_list.append(image_features.cpu().numpy())
                label_list.append(anno["category_id"] - 1)

        else:
            train_metafile_path = os.path.join(args.output_dir, f"{args.wandb_tag}_{args.clip_selection}_train.txt")
            train_metafile = open(train_metafile_path, "r")
            for line in train_metafile:
                image_path, label = line.strip().split(" ")
                image_path = image_path.split("@")[1]
                image = ext_train_zipfile.open(image_path)
                image = preprocess(Image.open(image)).unsqueeze(0).to(device)
                image_features = clip_model.encode_image(image)
                feat_list.append(image_features.cpu().numpy())
                label_list.append(int(label))

        ext_train_features = np.concatenate(feat_list, axis=0)
        ext_train_labels = np.array(label_list).reshape(-1, 1)

    if args.with_ext_train == "combined":
        train_features = np.concatenate([original_train_features, val_features, ext_train_features], axis=0)
        train_labels = np.concatenate([original_train_labels, val_labels, ext_train_labels], axis=0)
    elif args.with_ext_train == "original":
        train_features = np.concatenate([original_train_features, val_features], axis=0)
        train_labels = np.concatenate([original_train_labels, val_labels], axis=0)
    elif args.with_ext_train == "extended":
        train_features = ext_train_features
        train_labels = ext_train_labels

    logger.info(f"With Ext Train Mode: {args.with_ext_train}.")
    logger.info(f"Train size is {train_features.shape[0]}.")
    logger.info(f"Test size is {test_features.shape[0]}.")

    train_features = torch.from_numpy(train_features).to(device)
    test_features = torch.from_numpy(test_features).to(device)
    train_labels = torch.from_numpy(train_labels).to(device)
    test_labels = torch.from_numpy(test_labels).to(device)
    text_features = torch.from_numpy(text_features).to(device)

    train_features = F.normalize(train_features)
    test_features = F.normalize(test_features)
    if train_labels.squeeze().ndim == 1:
        train_labels_onehot = F.one_hot(train_labels.squeeze())
    else:
        train_labels_onehot = train_labels.squeeze()

    all_results = []
    fid = FID(feature=512)
    fid.update(train_features, real=True)
    fid.update(test_features, real=False)
    # TOP-K based approach
    test_train_sim = test_features @ train_features.T
    test_topk_train = test_train_sim.topk(config.RETRIEVAL.TOP_K, dim=-1).indices
    test_retrieval_topk_logits = (
        torch.gather(
            train_labels_onehot, 0, test_topk_train.view(-1)[:, None].expand(-1, train_labels_onehot.shape[-1])
        )
        .view(*test_topk_train.shape, -1)
        .type(text_features.dtype)
    )
    image_text_contrastive = (100.0 * test_features @ text_features).softmax(dim=-1)
    all_results.append(("image_text_contrastive", image_text_contrastive))
    logits_image_topk_label_mean = test_retrieval_topk_logits.mean(dim=1)
    all_results.append(("image_topk_label_mean", logits_image_topk_label_mean))
    logits_image_topk_label_mean_average_image_text_contrastive = (
        image_text_contrastive + logits_image_topk_label_mean
    ) / 2
    all_results.append(
        (
            "image_topk_label_mean_Average_image_text_contrastive",
            logits_image_topk_label_mean_average_image_text_contrastive,
        )
    )

    # Image CLS center based approach
    train_image_cls_centers = []
    if train_labels.shape[-1] == 1:
        for cls_idx in range(int(train_labels.max()) + 1):
            train_image_cls_centers.append(train_features[train_labels[:, 0] == cls_idx].mean(dim=0))
    else:
        for cls_idx in range(train_labels.shape[-1]):
            train_image_cls_centers.append(train_features[train_labels[:, cls_idx] == 1].mean(dim=0))
    train_image_cls_centers = torch.stack(train_image_cls_centers, dim=0)
    train_image_cls_centers = F.normalize(train_image_cls_centers)
    logits_image_imageCLSCenter_contrastive = (test_features @ train_image_cls_centers.T).softmax(dim=-1)
    all_results.append(("image_imageCLSCenter_contrastive", logits_image_imageCLSCenter_contrastive))
    mean_imageCLSCenter_text = F.normalize(text_features.T + train_image_cls_centers)
    logits_image_MeanTextImageCLSCenter_contrastive = (test_features @ mean_imageCLSCenter_text.T).softmax(dim=-1)
    all_results.append(("image_MeanTextImageCLSCenter_contrastive", logits_image_MeanTextImageCLSCenter_contrastive))
    for mode, logits in all_results:
        result = metric(test_labels.squeeze().cpu().detach().numpy(), logits.cpu().detach().numpy())
        logger.info(f"=> TEST {mode}: {metric.__name__} {100 * result:.3f}%")
        if wandb.run is not None:
            wandb.log({f"{mode}": 100 * result})

    logger.info(f"=> TEST fid: {float(fid.compute().cpu().numpy()):.3f}")
    wandb.log({"fid": float(fid.compute().cpu().numpy())})
    if wandb.run is not None:
        wandb.finish()


# dataset_list = [
#     "caltech-101",
#     "cifar-10",
#     "cifar-100",
#     "country211",
#     "dtd",
#     "eurosat_clip",
#     "fer-2013",
#     "fgvc-aircraft-2013b-variants102",
#     "food-101",
#     "gtsrb",
#     "hateful-memes",
#     "kitti-distance",
#     "mnist",
#     "oxford-flower-102",
#     "oxford-iiit-pets",
#     "patch-camelyon",
#     "rendered-sst2",
#     "resisc45_clip",
#     "stanford-cars",
#     "voc-2007-classification",
# ]


# import numpy as np
# from sklearn import svm
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler


# def proxy_a_distance(source_X, target_X, verbose=False):
#     """
#     Compute the Proxy-A-Distance of a source/target representation
#     """
#     nb_source = np.shape(source_X)[0]
#     nb_target = np.shape(target_X)[0]

#     if verbose:
#         print("PAD on", (nb_source, nb_target), "examples")

#     C_list = np.logspace(-5, 4, 10)

#     half_source, half_target = int(nb_source / 2), int(nb_target / 2)
#     train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
#     train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

#     test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
#     test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

#     best_risk = 1.0
#     for C in C_list:
#         clf = make_pipeline(StandardScaler(), svm.SVC(C=C, kernel="linear", verbose=False, gamma="auto"))
#         clf.fit(train_X, train_Y)

#         train_risk = np.mean(clf.predict(train_X) != train_Y)
#         test_risk = np.mean(clf.predict(test_X) != test_Y)

#         if verbose:
#             print("[ PAD C = %f ] train risk: %f  test risk: %f" % (C, train_risk, test_risk))

#         if test_risk > 0.5:
#             test_risk = 1.0 - test_risk

#         best_risk = min(best_risk, test_risk)

#     return 2 * (1.0 - 2 * best_risk)


# def feature_distances(args):
#     args.cfg = args.model
#     update_config(config, args)
#     config.defrost()
#     config.NAME = ""
#     config.freeze()
#     metric = get_metric(config.TEST.METRIC)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     clip_model, preprocess = clip.load("ViT-B/32", device=device)

#     ds_name = args.dataset_name

#     # for ds_name in dataset_list:
#     feat_list = []
#     logger.info("Preparing LAION400M images...")
#     image_list = glob.glob(
#         f"/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/sd21_generated_simple_template/{ds_name}/**/*.png",
#         recursive=True,
#     )
#     # image_list = glob.glob("/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/t2t_sep_prompt_30nn_archive/images/cifar-10/airplane/*.jpg", recursive=True)
#     selection_num = 200 if len(image_list) > 200 else len(image_list)
#     image_list = random.sample(image_list, selection_num)

#     logger.info("Extracting source LAION400M features...")
#     for img in image_list:
#         img = Image.open(img)
#         img = preprocess(img).unsqueeze(0).to(device)
#         feat = clip_model.encode_image(img)
#         feat_list.append(feat.detach().cpu().numpy())

#     args.cfg = f"./vision_benchmark/resources/datasets/{ds_name}.yaml"
#     update_config(config, args)

#     logger.info(f"Preparing {config.DATASET.DATASET} target features...")
#     source_features = np.concatenate(feat_list, axis=0)
#     all_features_and_labels, text_features = load_or_extract_features(args, config, test_split_only=False)
#     train_features, train_labels, val_features, val_labels, test_features, test_labels = all_features_and_labels
#     # target_features, target_labels = all_features_and_labels
#     source_features = F.normalize(torch.from_numpy(source_features).to(device)).cpu().numpy()
#     test_features = F.normalize(torch.from_numpy(test_features).to(device)).cpu().numpy()

#     logger.info(f"Source size is {source_features.shape[0]}.")
#     logger.info(f"Target size is {test_features.shape[0]}.")

#     proxy_distance = proxy_a_distance(source_features, test_features)
#     logger.info(f"Proxy distance from LAION400M to {config.DATASET.DATASET} is {proxy_distance}.")
