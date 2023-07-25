from PIL import ImageFile
from abc import ABC
import torch
from inspect import signature


def _identity(*args):
    return args


class _ImageOnlyTransform:
    def __init__(self, transform):
        self._transform = transform

    def __call__(self, image, boxes):
        return self._transform(image), boxes


class Dataset(torch.utils.data.Dataset, ABC):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        # Work around for corrupted files in datasets
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    @property
    def labels(self):
        """Returns a list of labels."""
        raise NotImplementedError

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, val):
        if not val:
            self._transform = _identity
        elif len(signature(val).parameters) == 1:
            self._transform = _ImageOnlyTransform(val)
        else:
            self._transform = val

    def close(self):
        """Release the resources allocated for this dataset."""
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
