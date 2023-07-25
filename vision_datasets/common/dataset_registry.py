import copy
import json
from .dataset_info import DatasetInfoFactory
from typing import Union


class DatasetRegistry:
    """
    A central registry of all available datasets
    """

    def __init__(self, datasets_json: Union[str, list]):
        if isinstance(datasets_json, list):
            self.datasets = [DatasetInfoFactory.create(d) for dj in datasets_json for d in json.loads(dj)]
        else:
            self.datasets = [DatasetInfoFactory.create(d) for d in json.loads(datasets_json)]

    def get_dataset_info(self, dataset_name, dataset_version=None):
        datasets = [d for d in self.datasets if d.name == dataset_name and (not dataset_version or d.version == dataset_version)]
        if not datasets:
            return None

        sorted_datasets = sorted(datasets, key=lambda d: d.version)
        return copy.deepcopy(sorted_datasets[-1])

    def list_data_version_and_types(self):
        return [{'name': d.name, 'version': d.version, 'type': d.type, 'description': d.description} for d in self.datasets]

    @staticmethod
    def _get_default_dataset_json(json_file_name):
        import sys
        py_version = sys.version_info
        if py_version.minor >= 7:
            import importlib.resources as pkg_resources
            from vision_datasets import resources
            datasets_json = pkg_resources.read_text(resources, json_file_name)
        else:
            import pkgutil
            resource_package = 'vision_datasets'
            resource_path = '/'.join(('resources', json_file_name))
            datasets_json = pkgutil.get_data(resource_package, resource_path)
        return datasets_json
