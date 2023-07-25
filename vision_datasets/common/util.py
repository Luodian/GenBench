import os
import pathlib
from typing import Union
import zipfile
from urllib import parse as urlparse
from urllib.parse import quote
from urllib.request import urlopen
from PIL import JpegImagePlugin
import json


def is_url(candidate: str):
    """
    necessary condition to be a url (not sufficient)
    Args:
        candidate (str):

    Returns:
        whether it could be a sas url or not

    """
    try:
        if not isinstance(candidate, str):
            return False

        result = urlparse.urlparse(candidate)
        return result.scheme and result.netloc
    except ValueError:
        return False


def write_to_json_file_utf8(dict, filepath: Union[str, pathlib.Path]):
    assert filepath

    pathlib.Path(filepath).write_text(json.dumps(dict, indent=2, ensure_ascii=False), encoding='utf-8')


class MultiProcessZipFile:
    """ZipFile which is readable from multi processes"""

    def __init__(self, filename):
        self.filename = filename
        self.zipfiles = {}

    def open(self, file):
        if os.getpid() not in self.zipfiles:
            self.zipfiles[os.getpid()] = zipfile.ZipFile(self.filename)
        return self.zipfiles[os.getpid()].open(file)

    def close(self):
        for z in self.zipfiles.values():
            z.close()
        self.zipfiles = {}

    def __getstate__(self):
        return {'filename': self.filename}

    def __setstate__(self, state):
        self.filename = state['filename']
        self.zipfiles = {}


class FileReader:
    """Reader to support <zip_filename>@<entry_name> style filename."""

    def __init__(self):
        self.zip_files = {}

    def open(self, name: Union[pathlib.Path, str], mode='r', encoding=None):
        name = str(name)
        # read file from url
        if is_url(name):
            return urlopen(self._encode_non_ascii(name))

        # read file from local zip: <zip_filename>@<entry_name>, e.g. images.zip@1.jpg
        if '@' in name:
            zip_path, file_path = name.split('@', 1)
            if zip_path not in self.zip_files:
                self.zip_files[zip_path] = MultiProcessZipFile(zip_path)
            return self.zip_files[zip_path].open(file_path)

        # read file from local dir
        return open(name, mode, encoding=encoding)

    def close(self):
        for zip_file in self.zip_files.values():
            zip_file.close()
        self.zip_files = {}

    @staticmethod
    def _encode_non_ascii(s):
        return ''.join([c if ord(c) < 128 else quote(c) for c in s])


def save_image_matching_quality(img, fp):
    """
    Save the image with mathcing qulaity, try not to compress
    https://stackoverflow.com/a/56675440/2612496
    """
    frmt = img.format

    if frmt == 'JPEG':
        quantization = getattr(img, 'quantization', None)
        subsampling = JpegImagePlugin.get_sampling(img)
        quality = 100 if quantization is None else -1
        img.save(fp, format=frmt, subsampling=subsampling, qtables=quantization, quality=quality)
    else:
        img.save(fp, format=frmt, quality=100)
