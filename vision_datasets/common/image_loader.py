import logging

from PIL import Image

logger = logging.getLogger(__name__)

# see https://exiv2.org/tags.html
ORIENTATION_EXIF_TAG = 0x0112


class PILImageLoader:
    """Load PIL image and fix image orientation using EXIF"""

    @staticmethod
    def load_from_stream(f):
        image = Image.open(f)
        img_format = image.format

        exif = image.getexif()
        orientation = exif.get(ORIENTATION_EXIF_TAG) if exif else None
        if orientation:
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.convert('RGB')
        image.format = img_format
        return image

    @staticmethod
    def load_from_file(filepath):
        try:
            with open(filepath, 'rb') as f:
                return PILImageLoader.load_from_stream(f)
        except Exception:
            logger.exception(f'Failed to load an image: {filepath}')
            raise
