import collections
import copy
import json
import logging
import os
import pathlib
import random
from typing import List, Dict, Union
from urllib import parse as urlparse
from PIL import Image
import numpy as np

from .constants import BBoxFormat, DatasetTypes, Formats
from .dataset_info import MultiTaskDatasetInfo
from .util import is_url, FileReader

logger = logging.getLogger(__name__)


def _unix_path(path: Union[pathlib.Path, str]):
    assert path is not None

    if isinstance(path, pathlib.Path):
        path = path.as_posix()

    return path.replace('\\', '/')


def _construct_full_path_generator(dirs: List[str]):
    """
    generate a function that appends dirs to a provided path, if dirs is empty, just return the path
    Args:
        dirs (str): dirs to be appended to a given path. None or empty str in dirs will be filtered.

    Returns:
        full_path_func: a func that appends dirs to a given path

    """
    dirs = [x for x in dirs if x]

    if dirs:
        def full_path_func(path: Union[pathlib.Path, str]):
            if isinstance(path, pathlib.Path):
                path = path.as_posix()
            to_join = [x for x in dirs + [path] if x]
            return _unix_path(os.path.join(*to_join))
    else:
        full_path_func = _unix_path

    return full_path_func


def _construct_full_url_generator(container_sas: str):
    if not container_sas:
        return _unix_path

    def add_path_to_url(url, path_or_dir):
        assert url

        if not path_or_dir:
            return url

        parts = urlparse.urlparse(url)
        path = _unix_path(os.path.join(parts[2], path_or_dir))
        url = urlparse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))

        return url

    def func(file_path):
        file_path = file_path.replace('.zip@', '/')  # cannot read from zip file with path targeting a url
        return add_path_to_url(container_sas, file_path)

    return func


def _construct_full_url_or_path_generator(container_sas_or_root_dir, prefix_dir=None):
    if container_sas_or_root_dir and is_url(container_sas_or_root_dir):
        return lambda path: _construct_full_url_generator(container_sas_or_root_dir)(_construct_full_path_generator([prefix_dir])(path))
    else:
        return lambda path: _construct_full_path_generator([container_sas_or_root_dir, prefix_dir])(path)


class ImageDataManifest:
    """
    Encapsulates the information and annotations of an image.

    img_path could be 1. a local path 2. a local path in a non-compressed zip file (`c:\a.zip@1.jpg`) or 3. a url.
    label_file_paths is a list of paths that have the same format with img_path
    """

    def __init__(self, id, img_path, width, height, labels, label_file_paths=None, labels_extra_info: dict = None):
        """
        Args:
            id (int or str): image id
            img_path (str): path to image
            width (int): image width
            height (int): image height
            labels (list or dict):
                classification: [c_id] for multiclass, [c_id1, c_id2, ...] for multilabel;
                detection: [[c_id, left, top, right, bottom], ...];
                image_caption: [caption1, caption2, ...];
                image_text_matching: [(text1, match (0 or 1), text2, match (0 or 1), ...)];
                multitask: dict[task, labels];
                image_matting: [mask1, mask2, ...], each mask is a 2D numpy array that has the same width and height with the image;
                image_regression: [target1].
            label_file_paths (list): list of paths of the image label files. "label_file_paths" only works for image matting task.
            labels_extra_info (dict[string, list]]): extra information about this image's labels
                Examples: 'iscrowd'
        """

        self.id = id
        self.img_path = img_path
        self.width = width
        self.height = height
        self._labels = labels
        self.label_file_paths = label_file_paths
        self.labels_extra_info = labels_extra_info or {}

    @property
    def labels(self):
        if self._labels:
            return self._labels
        elif self.label_file_paths:
            file_reader = FileReader()
            self._labels = []
            for label_file_path in self.label_file_paths:
                with file_reader.open(label_file_path) as f:
                    label = np.asarray(Image.open(f))
                    self._labels.append(label)
            file_reader.close()
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value


class DatasetManifest:
    """
    Encapsulates every information about a dataset including labelmap, images (width, height, path to image), and annotations. Information about each image is encapsulated in ImageDataManifest.
    """

    def __init__(self, images: List[ImageDataManifest], labelmap, data_type):
        """

        Args:
            images (list): image manifest
            labelmap (list or dict): labels, or labels by task name
            data_type (str or dict) : data type, or data type by task name

        """
        assert data_type != DatasetTypes.MULTITASK, 'For multitask, data_type should be a dict mapping task name to concrete data type.'

        if isinstance(labelmap, dict):
            assert isinstance(data_type, dict), 'labelmap being a dict indicating this is a multitask dataset, however the data_type is not a dict.'
            assert labelmap.keys() == data_type.keys(), f'mismatched task names in labelmap and task_type: {labelmap.keys()} vs {data_type.keys()}'

        self.images = images
        self.labelmap = labelmap
        self.data_type = data_type

        self._task_names = sorted(labelmap.keys()) if self.is_multitask else None

    @staticmethod
    def create_dataset_manifest(dataset_info, usage: str, container_sas_or_root_dir: str = None):

        if dataset_info.data_format == Formats.IRIS:
            return IrisManifestAdaptor.create_dataset_manifest(dataset_info, usage, container_sas_or_root_dir)
        if dataset_info.data_format == Formats.COCO:
            container_sas_or_root_dir = _construct_full_url_or_path_generator(container_sas_or_root_dir, dataset_info.root_folder)('')
            if dataset_info.type == DatasetTypes.MULTITASK:
                coco_file_by_task = {k: sub_taskinfo.index_files.get(usage) for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                data_type_by_task = {k: sub_taskinfo.type for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                return CocoManifestAdaptor.create_dataset_manifest(coco_file_by_task, data_type_by_task, container_sas_or_root_dir)

            return CocoManifestAdaptor.create_dataset_manifest(dataset_info.index_files.get(usage), dataset_info.type, container_sas_or_root_dir)

        raise RuntimeError(f'{dataset_info.data_format} not supported yet.')

    @property
    def is_multitask(self):
        """
        is this dataset multi-task dataset or not
        """

        return isinstance(self.data_type, dict)

    def __len__(self):
        return len(self.images)

    def _add_label_count(self, labels, n_images_by_class: list):
        if self.is_multitask:
            for task_name, task_labels in labels.items():
                for label in task_labels:
                    n_images_by_class[self._get_cid(label, task_name)] += 1
        else:
            for label in labels:
                n_images_by_class[self._get_cid(label)] += 1

    def _get_label_count(self, labels, n_images_by_class: list):
        if self.is_multitask:
            return [n_images_by_class[self._get_cid(label, task_name)] for task_name, task_labels in labels.items() for label in task_labels]
        else:
            return [n_images_by_class[self._get_cid(label)] for label in labels]

    def _get_cid(self, label, task_name=None):
        if task_name:  # multitask
            cnt = 0
            for t_name in self._task_names:
                if t_name == task_name:
                    break
                cnt += len(self.labelmap[t_name])

            return cnt + self._get_cid(label)
        elif isinstance(label, int):  # classification
            return label
        elif isinstance(label, list):  # detection
            return label[0]
        else:
            raise RuntimeError(f'unknown type of label: {type(label)}')

    def _is_negative(self, labels):
        n_labels = len(labels) if not self.is_multitask else sum([len(x) for x in labels.values()])
        return n_labels == 0

    def generate_coco_annotations(self):
        """
        Generate coco annotations, working for single task classification, detection, caption, and image regression only

        Returns:
            A dict of annotation data ready for coco json dump

        """

        images = []
        for i, x in enumerate(self.images):
            image = {'id': i + 1, 'file_name': x.img_path}
            if x.width:
                image['width'] = x.width
            if x.height:
                image['height'] = x.height
            images.append(image)

        annotations = []
        for img_id, img in enumerate(self.images):
            for ann in img.labels:
                coco_ann = {
                    'id': len(annotations) + 1,
                    'image_id': img_id + 1,
                }

                if DatasetTypes.is_classification(self.data_type):
                    coco_ann['category_id'] = ann + 1
                elif self.data_type == DatasetTypes.OD:
                    coco_ann['category_id'] = ann[0] + 1
                    coco_ann['bbox'] = [ann[1], ann[2], ann[3] - ann[1], ann[4] - ann[2]]
                elif self.data_type == DatasetTypes.IMCAP:
                    coco_ann['caption'] = ann
                elif self.data_type == DatasetTypes.IMAGE_REGRESSION:
                    coco_ann['target'] = ann
                else:
                    raise ValueError(f'Unsupported data type {self.data_type}')

                annotations.append(coco_ann)

        coco_dict = {'images': images, 'annotations': annotations}
        if self.data_type not in [DatasetTypes.IMCAP, DatasetTypes.IMAGE_REGRESSION]:
            coco_dict['categories'] = [{'id': i + 1, 'name': x} for i, x in enumerate(self.labelmap)]

        return coco_dict

    def train_val_split(self, train_ratio, random_seed=0):
        """
        Split the dataset into train and val set, with train set ratio being train_ratio.
        For multiclass dataset, the split ratio will be close to provided train_ratio, while for multilabel dataset, it is not guaranteed
        Multitask dataset and detection dataset are treated the same with multilabel dataset.
        Args:
            train_ratio(float): rough train set ratio, from 0 to 1
            random_seed: random seed

        Returns:
            train_manifest, val_manifest
        """
        if int(len(self.images) * train_ratio) == 0:
            return DatasetManifest([], self.labelmap, self.data_type), DatasetManifest(self.images, self.labelmap, self.data_type)

        if int(len(self.images) * train_ratio) == len(self.images):
            return DatasetManifest(self.images, self.labelmap, self.data_type), DatasetManifest([], self.labelmap, self.data_type)

        rng = random.Random(random_seed)
        images = list(self.images)
        rng.shuffle(images)

        train_imgs = []
        val_imgs = []
        n_train_imgs_by_class = [0] * len(self.labelmap) if not self.is_multitask else [0] * sum([len(x) for x in self.labelmap.values()])
        n_val_imgs_by_class = [0] * len(self.labelmap) if not self.is_multitask else [0] * sum([len(x) for x in self.labelmap.values()])
        test_train_ratio = (1 - train_ratio) / train_ratio
        n_train_neg = 0
        n_val_neg = 0
        for image in images:
            if self._is_negative(image.labels):
                if n_train_neg == 0 or n_val_neg / n_train_neg >= test_train_ratio:
                    n_train_neg += 1
                    train_imgs.append(image)
                else:
                    n_val_neg += 1
                    val_imgs.append(image)

                continue

            train_cnt = self._get_label_count(image.labels, n_train_imgs_by_class)
            val_cnt = self._get_label_count(image.labels, n_val_imgs_by_class)
            train_cnt_sum = sum(train_cnt) * test_train_ratio
            train_cnt_min = min(train_cnt) * test_train_ratio
            val_cnt_sum = sum(val_cnt)
            val_cnt_min = min(val_cnt)
            if val_cnt_min < train_cnt_min or (val_cnt_min == train_cnt_min and val_cnt_sum < train_cnt_sum):
                val_imgs.append(image)
                self._add_label_count(image.labels, n_val_imgs_by_class)
            else:
                train_imgs.append(image)
                self._add_label_count(image.labels, n_train_imgs_by_class)

        return DatasetManifest(train_imgs, self.labelmap, self.data_type), DatasetManifest(val_imgs, self.labelmap, self.data_type)

    def sample_categories(self, category_indices: List):
        """
        Sample a new dataset of selected categories. Works for single IC and OD dataset only.
        Args:
            category_indices: indices of the selected categories

        Returns:
            a sampled dataset with selected categories

        """

        assert self.data_type in [DatasetTypes.IC_MULTICLASS, DatasetTypes.IC_MULTILABEL, DatasetTypes.OD]
        assert category_indices
        assert max(category_indices) < len(self.labelmap)

        category_id_remap = {o_cid: n_cid for n_cid, o_cid in enumerate(category_indices)}
        new_labelmap = [self.labelmap[x] for x in category_indices]
        new_images = []
        for img in self.images:
            new_img = copy.deepcopy(img)
            if DatasetTypes.is_classification(self.data_type):
                new_img.labels = [category_id_remap[x] for x in new_img.labels if x in category_id_remap]
            else:
                new_img.labels = [[category_id_remap[x[0]], x[1], x[2], x[3], x[4]] for x in img.labels if x[0] in category_id_remap]

            new_images.append(new_img)
        return DatasetManifest(new_images, new_labelmap, self.data_type)

    def sample_subset(self, num_samples, with_replacement=False, random_seed=0):
        """
        Sample a subset of num_samples images. When with_replacement is False and num_samples is larger than the dataset, the whole dataset will be returned
        Args:
            num_samples (int): number of images to be sampled
            with_replacement (bool): with replacement or not
            random_seed (int): random seed

        Returns:
            a sampled dataset
        """

        rnd = random.Random(random_seed)
        if not with_replacement:
            if num_samples >= len(self.images):
                sampled_images = self.images
            else:
                sampled_images = rnd.sample(self.images, num_samples)
        else:
            sampled_images = [rnd.choice(self.images) for _ in range(num_samples)]

        sampled_images = [copy.deepcopy(x) for x in sampled_images]
        return DatasetManifest(sampled_images, self.labelmap, self.data_type)

    def sample_few_shot_subset(self, num_samples_per_class, random_seed=0):
        """
        Sample a few-shot dataset, with the number of images per class below num_samples_per_class.
        For multiclass dataset, this is always possible, while for multilabel dataset, it is not guaranteed
        Multitask dataset and detection dataset are treated the same with multilabel dataset.

        This method tries to get balanced results.

        Note that negative images will be added to the subset up to num_samples_per_class.

        Args:
            num_samples_per_class: rough number samples per class to sample
            random_seed: random seed

        Returns:
            a sampled few-shot subset
        """

        assert num_samples_per_class > 0

        sampled_images = []
        rng = random.Random(random_seed)
        images = list(self.images)
        rng.shuffle(images)
        n_imgs_by_class = [0] * len(self.labelmap) if not self.is_multitask else [0] * sum([len(x) for x in self.labelmap.values()])
        neg_img_cnt = 0
        for image in images:
            if self._is_negative(image.labels):
                if neg_img_cnt < num_samples_per_class:
                    neg_img_cnt += 1
                    sampled_images.append(image)
                continue

            img_label_cnt = self._get_label_count(image.labels, n_imgs_by_class)

            if min(img_label_cnt) >= num_samples_per_class:
                continue

            if min(img_label_cnt) <= num_samples_per_class / 2 or max(img_label_cnt) <= 1.5 * num_samples_per_class:
                sampled_images.append(image)
                self._add_label_count(image.labels, n_imgs_by_class)

            if min(n_imgs_by_class) >= num_samples_per_class:
                break

        sampled_images = [copy.deepcopy(x) for x in sampled_images]
        return DatasetManifest(sampled_images, self.labelmap, self.data_type)

    def sample_subset_by_ratio(self, sampling_ratio):
        """
        Sample a dataset so that each labels appears by at least the given sampling_ratio. In case of multiclass dataset, the number of sampled images will be N * sampling_ratio.
        For multilabel or object detection datasets, the total number of images will be bigger than that.

        Args:
            sampling_ratio (float): sampling ratio. must be 0 < x < 1.

        Returns:
            A sampled dataset (DatasetManifest)
        """
        assert 0 < sampling_ratio < 1

        if self.is_multitask:
            labels = [[self._get_cid(c, t) for t, t_labels in image.labels.items() for c in t_labels] for image in self.images]
        else:
            labels = [[self._get_cid(c) for c in image.labels] for image in self.images]

        # Create a dict {label_id: [image_id, ...], ...}
        # Note that image_id can be included multiple times if the dataset is multilabel, objectdetection, or multitask.
        label_image_map = collections.defaultdict(list)
        for i, image_labels in enumerate(labels):
            if not image_labels:
                label_image_map[-1].append(i)
            for label in image_labels:
                label_image_map[label].append(i)

        # From each lists, sample max(1, N * ratio) images.
        sampled_image_ids = set()
        for image_ids in label_image_map.values():
            sampled_image_ids |= set(random.sample(image_ids, max(1, int(len(image_ids) * sampling_ratio))))

        sampled_images = [copy.deepcopy(self.images[i]) for i in sampled_image_ids]
        return DatasetManifest(sampled_images, self.labelmap, self.data_type)

    def sample_few_shots_subset_greedy(self, num_min_samples_per_class, random_seed=0):
        """Greedy few-shots sampling method.
        Randomly pick images from the original datasets until all classes have at least {num_min_images_per_class} tags/boxes.

        Note that images without any tag/box will be ignored. All images in the subset will have at least one tag/box.

        Args:
            num_min_samples_per_class (int): The minimum number of samples per class.
            random_seed (int): Random seed to use.

        Returns:
            A samped dataset (DatasetManifest)

        Raises:
            RuntimeError if it couldn't find num_min_samples_per_class samples for all classes
        """

        assert num_min_samples_per_class > 0
        images = list(self.images)
        rng = random.Random(random_seed)
        rng.shuffle(images)

        num_classes = len(self.labelmap) if not self.is_multitask else sum(len(x) for x in self.labelmap.values())
        total_counter = collections.Counter({i: num_min_samples_per_class for i in range(num_classes)})
        sampled_images = []
        for image in images:
            counts = collections.Counter([self._get_cid(c) for c in image.labels] if not self.is_multitask else [self._get_cid(c, t) for t, t_labels in image.labels.items() for c in t_labels])
            if set((+total_counter).keys()) & set(counts.keys()):
                total_counter -= counts
                sampled_images.append(image)

            if not +total_counter:
                break

        if +total_counter:
            raise RuntimeError(f"Couldn't find {num_min_samples_per_class} samples for some classes: {+total_counter}")

        sampled_images = [copy.deepcopy(x) for x in sampled_images]
        return DatasetManifest(sampled_images, self.labelmap, self.data_type)

    def spawn(self, num_samples, random_seed=0, instance_weights: List = None):
        """Spawn manifest to a size.
        To ensure each class has samples after spawn, we first keep a copy of original data, then merge with sampled data.
        If instance_weights is not provided, spawn follows class distribution.
        Otherwise spawn the dataset so that the instances follow the given weights. In this case the spawned size is not guranteed to be num_samples.

        Args:
            num_samples (int): size of spawned manifest. Should be larger than the current size.
            random_seed (int): Random seed to use.
            instance_weights (list): weight of each instance to spawn, >= 0.

        Returns:
            Spawned dataset (DatasetManifest)
        """
        assert num_samples > len(self)
        if instance_weights is not None:
            assert len(instance_weights) == len(self)
            assert all([x >= 0 for x in instance_weights])

            sum_weights = sum(instance_weights)
            # Distribute the number of num_samples to each image by the weights. The original image is subtracted.
            n_copies_per_sample = [max(0, round(w / sum_weights * num_samples - 1)) for w in instance_weights]
            spawned_images = []
            for image, n_copies in zip(self.images, n_copies_per_sample):
                spawned_images += [copy.deepcopy(image) for _ in range(n_copies)]

            sampled_manifest = DatasetManifest(spawned_images, self.labelmap, self.data_type)
        else:
            sampled_manifest = self.sample_subset(num_samples - len(self), with_replacement=True, random_seed=random_seed)

        # Merge with the copy of the original dataset to ensure each class has sample.
        return DatasetManifest.merge(self, sampled_manifest, flavor=0)

    @staticmethod
    def merge(*args, flavor: int = 0):
        """
        merge multiple data manifests into one.

        Args:
            args: manifests to be merged
            flavor: flavor of dataset merge (not difference for captioning)
                0: merge manifests of the same type and the same labelmap (for multitask, it should be same set of tasks and same labelmap for each task)
                1: concat manifests of the same type, the new labelmap are concats of all labelmaps in all manifest (for multitask, duplicate task names are not allowed)
        """

        assert len(args) >= 1, 'less than one manifests provided, not possible to merged.'
        assert all([arg is not None for arg in args]), '"None" manifest found'

        args = [arg for arg in args if arg]
        if len(args) == 1:
            logger.warning('Only one manifest provided. Nothing to be merged.')
            return args[0]

        if any([isinstance(x.data_type, dict) for x in args]):
            assert all([isinstance(x.data_type, dict) for x in args]), 'Cannot merge multitask manifest and single task manifest'
        else:
            assert len(set([x.data_type for x in args])) == 1, 'All manifests must be of the same data type'

        if flavor == 0:
            return DatasetManifest._merge_with_same_labelmap(*args)
        elif flavor == 1:
            return DatasetManifest._merge_with_concat(*args)
        else:
            raise ValueError(f'Unknown flavor {flavor}.')

    @staticmethod
    def _merge_with_same_labelmap(*args):
        for i in range(len(args)):
            if i > 0 and args[i].labelmap != args[i - 1].labelmap:
                raise ValueError('labelmap must be the same for all manifests.')
            if i > 0 and args[i].data_type != args[i - 1].data_type:
                raise ValueError('Data type must be the same for all manifests.')

        images = [y for x in args for y in x.images]

        return DatasetManifest(images, args[0].labelmap, args[0].data_type)

    @staticmethod
    def _merge_with_concat(*args):
        data_type = args[0].data_type

        if data_type in [DatasetTypes.IMCAP, DatasetTypes.IMAGE_REGRESSION]:
            return DatasetManifest._merge_with_same_labelmap(args)

        if isinstance(data_type, dict):  # multitask
            labelmap = {}
            data_types = {}
            for manifest in args:
                for k, v in manifest.labelmap.items():
                    if k in labelmap:
                        raise ValueError(f'Failed to merge dataset manifests, as due to task with name {k} exists in more than one manifest.')

                    labelmap[k] = v

                for k, v in manifest.data_type.items():
                    data_types[k] = v

            return DatasetManifest([y for x in args for y in x.images], labelmap, data_types)

        labelmap = []
        images = []

        for manifest in args:
            label_offset = len(labelmap)
            for img_manifest in manifest.images:
                new_img_manifest = copy.deepcopy(img_manifest)
                if DatasetTypes.is_classification(data_type):
                    new_img_manifest.labels = [x + label_offset for x in new_img_manifest.labels]
                elif data_type == DatasetTypes.OD:
                    for label in new_img_manifest.labels:
                        label[0] += label_offset
                else:
                    raise ValueError(f'Unsupported type in merging {data_type}')

                images.append(new_img_manifest)
            labelmap.extend(manifest.labelmap)

        return DatasetManifest(images, labelmap, data_type)

    @staticmethod
    def create_multitask_manifest(manifest_by_task: dict):
        """
        Merge several manifests into a multitask dataset in a naive way, assuming images from different manifests are independent different images.
        Args:
            manifest_by_task (dict): manifest by task name

        Returns:
            a merged multitask manifest
        """

        task_names = sorted(list(manifest_by_task.keys()))
        images = []
        for task_name in task_names:
            for img in manifest_by_task[task_name].images:
                new_img = copy.deepcopy(img)
                new_img.labels = {task_name: new_img.labels}
                images.append(new_img)

        labelmap = {task_name: manifest_by_task[task_name].labelmap for task_name in task_names}
        data_types = {task_name: manifest_by_task[task_name].data_type for task_name in task_names}

        return DatasetManifest(images, labelmap, data_types)


def _generate_multitask_dataset_manifest(manifest_by_task: Dict[str, DatasetManifest]):
    images_by_id = {}
    for task_name, task_manifest in manifest_by_task.items():
        if not task_manifest:
            continue
        for image in task_manifest.images:
            if image.id not in images_by_id:
                multi_task_image_manifest = ImageDataManifest(image.id, image.img_path, image.width, image.height, {task_name: image.labels})
                images_by_id[image.id] = multi_task_image_manifest
            else:
                images_by_id[image.id].labels[task_name] = image.labels

    if not images_by_id:
        return None

    labelmap_by_task = {k: manifest.labelmap for k, manifest in manifest_by_task.items()}
    dataset_types_by_task = {k: manifest.data_type for k, manifest in manifest_by_task.items()}
    return DatasetManifest([v for v in images_by_id.values()], labelmap_by_task, dataset_types_by_task)


class IrisManifestAdaptor:
    """
    Adaptor for generating dataset manifest from iris format
    """

    @staticmethod
    def create_dataset_manifest(dataset_info, usage: str, container_sas_or_root_dir: str = None):
        """

        Args:
            dataset_info (MultiTaskDatasetInfo or .DatasetInfo):  dataset info
            usage (str): which usage of data to construct
            container_sas_or_root_dir (str): sas url if the data is store in a azure blob container, or a local root dir
        """
        assert dataset_info
        assert usage

        if dataset_info.type in [DatasetTypes.IMCAP, DatasetTypes.IMAGE_TEXT_MATCHING, DatasetTypes.IMAGE_MATTING, DatasetTypes.IMAGE_REGRESSION]:
            raise ValueError(f'Iris format is not supported for {dataset_info.type} task, please use COCO format!')
        if isinstance(dataset_info, MultiTaskDatasetInfo):
            dataset_manifest_by_task = {k: IrisManifestAdaptor.create_dataset_manifest(task_info, usage, container_sas_or_root_dir) for k, task_info in dataset_info.sub_task_infos.items()}
            return _generate_multitask_dataset_manifest(dataset_manifest_by_task)
        if usage not in dataset_info.index_files:
            return None

        file_reader = FileReader()

        dataset_info = copy.deepcopy(dataset_info)
        get_full_sas_or_path = _construct_full_url_or_path_generator(container_sas_or_root_dir, dataset_info.root_folder)

        max_index = 0
        labelmap = None
        if not dataset_info.labelmap:
            logger.warning(f'{dataset_info.name}: labelmap is missing!')
        else:
            # read tag names
            with file_reader.open(get_full_sas_or_path(dataset_info.labelmap), encoding='utf-8') as file_in:
                labelmap = [IrisManifestAdaptor._purge_line(line) for line in file_in if IrisManifestAdaptor._purge_line(line) != '']

        # read image width and height
        img_wh = None
        if dataset_info.image_metadata_path:
            img_wh = IrisManifestAdaptor._load_img_width_and_height(file_reader, get_full_sas_or_path(dataset_info.image_metadata_path))

        # read image index files
        images = []
        with file_reader.open(get_full_sas_or_path(dataset_info.index_files[usage])) as file_in:
            for line in file_in:
                line = IrisManifestAdaptor._purge_line(line)
                if not line:
                    continue
                parts = line.rsplit(' ', maxsplit=1)  # assumption: only the image file path can have spaces
                img_path = parts[0]
                # if dataset_info.name.endswith('-generated') and usage == 'train':
                #     img_path = 'generated/' + img_path
                label_or_label_file = parts[1] if len(parts) == 2 else None

                w, h = img_wh[img_path] if img_wh else (None, None)
                if DatasetTypes.is_classification(dataset_info.type):
                    img_labels = [int(x) for x in label_or_label_file.split(',')] if label_or_label_file else []
                else:
                    img_labels = IrisManifestAdaptor._load_detection_labels_from_file(file_reader, get_full_sas_or_path(label_or_label_file)) if label_or_label_file else []

                if not labelmap and img_labels:
                    c_indices = [x[0] for x in img_labels] if isinstance(img_labels[0], list) else img_labels
                    max_index = max(max(c_indices), max_index)

                images.append(ImageDataManifest(img_path, get_full_sas_or_path(img_path), w, h, img_labels))

            if not labelmap:
                labelmap = [str(x) for x in range(max_index + 1)]
            file_reader.close()
        return DatasetManifest(images, labelmap, dataset_info.type)

    @staticmethod
    def _load_img_width_and_height(file_reader, file_path):
        img_wh = dict()
        with file_reader.open(file_path) as file_in:
            for line in file_in:
                line = IrisManifestAdaptor._purge_line(line)
                if line == '':
                    continue
                location, w, h = line.split()
                img_wh[location] = (int(w), int(h))

        return img_wh

    @staticmethod
    def _load_detection_labels_from_file(file_reader, image_label_file_path):

        with file_reader.open(image_label_file_path) as label_in:
            label_lines = [IrisManifestAdaptor._purge_line(line) for line in label_in]

        img_labels = []
        for label_line in label_lines:
            parts = label_line.split()

            assert len(parts) == 5  # regions
            box = [float(p) for p in parts]
            box[0] = int(box[0])
            img_labels.append(box)

        return img_labels

    @staticmethod
    def _purge_line(line):
        if not isinstance(line, str):
            line = line.decode('utf-8')

        return line.strip()


class CocoManifestAdaptor:
    """
    Adaptor for generating manifest from coco format

    image paths should be stored under 'file_name'
    """

    @staticmethod
    def create_dataset_manifest(coco_file_path_or_url: Union[str, dict, pathlib.Path], data_type, container_sas_or_root_dir: str = None):
        """ construct a dataset manifest out of coco file
        Args:
            coco_file_path_or_url (str or pathlib.Path or dict): path or url to coco file. dict if multitask
            data_type (str or dict): type of dataset. dict if multitask
            container_sas_or_root_dir (str): container sas if resources are store in blob container, or a local dir
        """

        if not coco_file_path_or_url:
            return None

        assert data_type

        if isinstance(coco_file_path_or_url, dict):
            assert isinstance(data_type, dict)
            dataset_manifest_by_task = {k: CocoManifestAdaptor.create_dataset_manifest(coco_file_path_or_url[k], data_type[k], container_sas_or_root_dir)
                                        for k in coco_file_path_or_url}

            return _generate_multitask_dataset_manifest(dataset_manifest_by_task)

        get_full_sas_or_path = _construct_full_url_or_path_generator(container_sas_or_root_dir)

        file_reader = FileReader()
        # read image index files
        coco_file_path_or_url = coco_file_path_or_url if is_url(coco_file_path_or_url) else get_full_sas_or_path(coco_file_path_or_url)
        with file_reader.open(coco_file_path_or_url, encoding='utf-8') as file_in:
            coco_manifest = json.load(file_in)

        file_reader.close()

        def get_file_path(info_dict: dict, file_name):
            zip_prefix = info_dict.get('zip_file', '')
            if zip_prefix:
                zip_prefix += '@'

            return get_full_sas_or_path(zip_prefix + file_name)

        images_by_id = {img['id']: ImageDataManifest(img['id'], get_file_path(img, img['file_name']), img.get('width'), img.get('height'), [], {}) for img in coco_manifest['images']}
        process_labels_without_categories = None
        if data_type == DatasetTypes.IMCAP:
            def process_labels_without_categories(image):
                image.labels.append(annotation['caption'])
        elif data_type == DatasetTypes.IMAGE_TEXT_MATCHING:
            def process_labels_without_categories(image):
                image.labels.append((annotation['text'], annotation['match']))
        elif data_type == DatasetTypes.IMAGE_MATTING:
            def process_labels_without_categories(image):
                image.label_file_paths = image.label_file_paths or []
                image.label_file_paths.append(get_file_path(annotation, annotation['label']))
        elif data_type == DatasetTypes.IMAGE_REGRESSION:
            def process_labels_without_categories(image):
                assert len(image.labels) == 0, f"There should be exactly one label per image for image_regression datasets, but image with id {annotation['image_id']} has more than one"
                image.labels.append(annotation['target'])

        if process_labels_without_categories:
            for annotation in coco_manifest['annotations']:
                process_labels_without_categories(images_by_id[annotation['image_id']])
            images = [x for x in images_by_id.values()]
            return DatasetManifest(images, None, data_type)

        cate_id_name = [(cate['id'], cate['name']) for cate in coco_manifest['categories']]
        cate_id_name.sort(key=lambda x: x[0])
        label_id_to_pos = {x[0]: i for i, x in enumerate(cate_id_name)}
        labelmap = [x[1] for x in cate_id_name]

        bbox_format = coco_manifest.get('bbox_format', BBoxFormat.LTWH)
        BBoxFormat.validate(bbox_format)

        for annotation in coco_manifest['annotations']:
            c_id = label_id_to_pos[annotation['category_id']]
            img = images_by_id[annotation['image_id']]
            if 'bbox' in annotation:
                bbox = annotation['bbox']
                if bbox_format == BBoxFormat.LTWH:
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                label = [c_id] + bbox
                img.labels_extra_info['iscrowd'] = img.labels_extra_info.get('iscrowd', [])
                img.labels_extra_info['iscrowd'].append(annotation.get('iscrowd', 0))
            else:
                label = c_id

            img.labels.append(label)

        images = [x for x in images_by_id.values()]
        images.sort(key=lambda x: x.id)

        return DatasetManifest(images, labelmap, data_type)
