# Compute A-distance using numpy and sklearn
# Reference: Analysis of representations in domain adaptation, NIPS-07.

import numpy as np
from sklearn import svm
from collections import defaultdict
from vision_benchmark.datasets import class_map, template_map
import zipfile
import os
from vision_benchmark.common.constants import get_dataset_hub, VISION_DATASET_STORAGE

import random
from PIL import Image
from common_utils import process_namings
import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> | <cyan>{name}</cyan><cyan>:line {line}</cyan>: <level>{message}</level>", level="INFO")

def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)

def prepare_features(dataset_name):
    concept_list = class_map[dataset_name]

    dataset_hub = get_dataset_hub()
    from vision_datasets import Usages
    
    manifest = dataset_hub.create_dataset_manifest(VISION_DATASET_STORAGE, local_dir="/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/datasets/", name=dataset_name, usage=Usages.TRAIN_PURPOSE)
    random_seed = 1337
    dataset_info = manifest[1]
    dataset_manifest = manifest[0].sample_few_shot_subset(20, random_seed)
    category_list = defaultdict(list)
    for image in dataset_manifest.images:
        try:
            cate_name = concept_list[image.labels[0]]
            image_tuple = (image.img_path, image.labels[0])
            category_list[cate_name].append(image_tuple)
        except:
            continue

    train_zipfile_path = dataset_manifest.images[0].img_path.split("@")[0]
    train_zipfile = zipfile.ZipFile(train_zipfile_path, "r")
    ref_shot = args.ref_shot if args.ref_shot else 5
    # train_zip_infolist = train_zipfile.infolist()
    for concept in concept_list:
        logger.info(f"Processing concept {concept} ...")
        saved_concept_dir = process_namings(concept)
        ref_concept_dir = os.path.join(args.output_dir, f"{saved_concept_dir}_instance_dir")
        os.makedirs(ref_concept_dir, exist_ok=True)

        # select 5 images from the concept as instance images
        ref_shot = ref_shot if len(category_list[concept]) >= ref_shot else len(category_list[concept])
        selected_instance = random.sample(category_list[concept], ref_shot)
        for idx, (img_path, label) in enumerate(selected_instance):
            img_file = train_zipfile.open(img_path.split("@")[1])
            img = Image.open(img_file)
            img.save(os.path.join(ref_concept_dir, f"{idx}.png"))
            logger.info(f"Saved image {idx} to {ref_concept_dir}")

if __name__ == "__main__":
    source_X = np.random.rand(100, 100)
    target_X = np.random.rand(100, 100)
    print(proxy_a_distance(source_X, target_X, verbose=True))