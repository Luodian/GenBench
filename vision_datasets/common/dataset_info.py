from .constants import Usages, DatasetTypes, Formats


class DatasetInfoFactory:
    @staticmethod
    def create(data_info_dict):
        if DatasetInfoFactory.is_multitask(data_info_dict['type']):
            return MultiTaskDatasetInfo(data_info_dict)
        return DatasetInfo(data_info_dict)

    @staticmethod
    def is_multitask(task_type):
        return 'multitask' in task_type


class BaseDatasetInfo:
    """
    Info fields common to both all datasets regardless of whether it is coco or iris, single task or multitask
    """

    def __init__(self, dataset_info_dict):
        self.name = dataset_info_dict['name']
        self.version = dataset_info_dict.get('version', 1)
        self.type = dataset_info_dict['type']
        self.root_folder = dataset_info_dict.get('root_folder')
        self.description = dataset_info_dict.get('description', '')
        self.data_format = dataset_info_dict.get('format', Formats.IRIS)

from collections import defaultdict
class DatasetInfo(BaseDatasetInfo):

    def __init__(self, dataset_info_dict):
        data_type = dataset_info_dict.get('type')
        assert data_type in DatasetTypes.VALID_TYPES, f'Unknown type {data_type}. Valid types are {DatasetTypes.VALID_TYPES}.'
        assert not DatasetInfoFactory.is_multitask(dataset_info_dict['type'])
        super(DatasetInfo, self).__init__(dataset_info_dict)

        self.index_files = dict()
        self.files_for_local_usage = defaultdict(dict)
        for usage in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
            if usage in dataset_info_dict:
                self.index_files[usage] = dataset_info_dict[usage]['index_path']
                self.files_for_local_usage[usage] = dataset_info_dict[usage].get('files_for_local_usage', [])

        # Below are needed for iris format only. As both image h and w and labelmaps are included in the coco annotation files
        self.labelmap = dataset_info_dict.get('labelmap')
        self.image_metadata_path = dataset_info_dict.get('image_metadata_path')

    @property
    def train_path(self):
        return self.index_files[Usages.TRAIN_PURPOSE] if Usages.TRAIN_PURPOSE in self.index_files else None

    @property
    def val_path(self):
        return self.index_files[Usages.VAL_PURPOSE] if Usages.VAL_PURPOSE in self.index_files else None

    @property
    def test_path(self):
        return self.index_files[Usages.TEST_PURPOSE] if Usages.TEST_PURPOSE in self.index_files else None

    @property
    def train_support_files(self):
        """Path to the files which are referenced by the train dataset file"""

        return self.files_for_local_usage[Usages.TRAIN_PURPOSE] if Usages.TRAIN_PURPOSE in self.index_files else []

    @property
    def val_support_files(self):
        """Path to the files which are referenced by the validation dataset file"""

        return self.files_for_local_usage[Usages.VAL_PURPOSE] if Usages.VAL_PURPOSE in self.index_files else []

    @property
    def test_support_files(self):
        """Path to the files which are referenced by the test dataset file"""

        return self.files_for_local_usage[Usages.TEST_PURPOSE] if Usages.TEST_PURPOSE in self.index_files else []


class MultiTaskDatasetInfo(BaseDatasetInfo):
    def __init__(self, dataset_info_dict):
        assert 'tasks' in dataset_info_dict
        assert DatasetInfoFactory.is_multitask(dataset_info_dict['type'])

        super(MultiTaskDatasetInfo, self).__init__(dataset_info_dict)

        tasks = dataset_info_dict['tasks']
        info_dict = {}
        for task_name, task_info in tasks.items():
            info_dict[task_name] = DatasetInfo({**dataset_info_dict, **task_info})

        self.sub_task_infos = info_dict

    @property
    def task_names(self):
        return list(self.sub_task_infos.keys())

    def get_task_dataset_info(self, task_name: str):
        return self.sub_task_infos[task_name]

    @property
    def train_support_files(self):
        """Path to the files which are referenced by the train dataset file"""
        return list(set([x for task_info in self.sub_task_infos.values() for x in task_info.train_support_files]))

    @property
    def val_support_files(self):
        """Path to the files which are referenced by the validation dataset file"""

        return list(set([x for task_info in self.sub_task_infos.values() for x in task_info.val_support_files]))

    @property
    def test_support_files(self):
        """Path to the files which are referenced by the validation dataset file"""

        return list(set([x for task_info in self.sub_task_infos.values() for x in task_info.test_support_files]))
