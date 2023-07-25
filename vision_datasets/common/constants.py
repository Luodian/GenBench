class DatasetTypes:
    IC_MULTILABEL = 'classification_multilabel'
    IC_MULTICLASS = 'classification_multiclass'
    OD = 'object_detection'
    MULTITASK = 'multitask'
    IMCAP = 'image_caption'
    IMAGE_TEXT_MATCHING = 'image_text_matching'
    IMAGE_MATTING = 'image_matting'
    IMAGE_REGRESSION = 'image_regression'
    VALID_TYPES = [IC_MULTILABEL, IC_MULTICLASS, OD, MULTITASK, IMCAP, IMAGE_TEXT_MATCHING, IMAGE_MATTING, IMAGE_REGRESSION]

    @staticmethod
    def is_classification(dataset_type):
        return dataset_type.startswith('classification')


class Usages:
    TRAIN_PURPOSE = 'train'
    VAL_PURPOSE = 'val'
    TEST_PURPOSE = 'test'


class Formats:
    IRIS = 'iris'
    COCO = 'coco'


class BBoxFormat:
    LTRB = 'ltrb'
    LTWH = 'ltwh'

    VALID_TYPES = [LTRB, LTWH]

    @staticmethod
    def validate(bbox_format):
        assert bbox_format in BBoxFormat.VALID_TYPES, f'Invalid bbox_format: {bbox_format}. Must be {BBoxFormat.VALID_TYPES}'
