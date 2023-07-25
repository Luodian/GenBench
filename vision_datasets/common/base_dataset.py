from abc import ABC, abstractmethod

from .dataset_info import BaseDatasetInfo


class BaseDataset(ABC):
    def __init__(self, dataset_info: BaseDatasetInfo):

        self.dataset_info = dataset_info

    def __getitem__(self, idx):
        """ iterate through the dataset

        Args:
            idx: can be a single index or range

        Returns:
            requested sample(s)
        """

        if isinstance(idx, int):
            if idx >= self.__len__():
                raise IndexError
            return self._get_single_item(idx)

        stop = min(self.__len__(), idx.stop)
        return [self.__getitem__(i) for i in range(idx.start, stop, idx.step)] if idx.step else [self.__getitem__(i) for i in range(idx.start, stop)]

    @property
    @abstractmethod
    def labels(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def _get_single_item(self, index):
        pass

    @abstractmethod
    def close(self):
        """ release resources
        """
        pass
