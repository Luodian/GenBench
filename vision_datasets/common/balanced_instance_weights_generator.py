"""Generate instance weights from DatasetManifest, which can be used for balancing the dataset by sampling instances based on the weights. Only works for classification, detection, multitask."""

import logging
from collections import Counter

import numpy

logger = logging.getLogger(__name__)


class BalancedInstanceWeightsGenerator(object):
    NEG_CLASS_INDEX = -1

    @staticmethod
    def generate(data_manifest, soft=True, weight_upper=5, weight_lower=0.2):
        assert data_manifest is not None

        logger.info("Generating instance weights for dataset balancing.")
        image_tags = [BalancedInstanceWeightsGenerator._process_tags(x.labels) for x in data_manifest.images]

        class_wise_image_counter = Counter()
        for tags in image_tags:
            class_wise_image_counter.update(tags)

        mean_class_wise_image_tag_count = numpy.mean(list(class_wise_image_counter.values()))
        class_wise_multipliers = {x: mean_class_wise_image_tag_count / class_wise_image_counter[x] for x in class_wise_image_counter}
        if soft:
            class_wise_multipliers = {x: numpy.sqrt(class_wise_multipliers[x]) for x in class_wise_multipliers}

        class_wise_multipliers = {x: BalancedInstanceWeightsGenerator._scope_multiplier(class_wise_multipliers[x], weight_upper, weight_lower) for x in class_wise_multipliers}

        image_weights = [BalancedInstanceWeightsGenerator._get_instance_multiplier(tags, class_wise_multipliers, weight_upper, weight_lower) for tags in image_tags]

        logger.info(f'instance weights: max {max(image_weights)}, min {min(image_weights)}, len {len(image_weights)}')
        return image_weights

    @staticmethod
    def _process_tags(tags):
        if not tags:
            return [BalancedInstanceWeightsGenerator.NEG_CLASS_INDEX]

        if isinstance(tags, dict):
            return list(set(f'{str(k)}_{str(v[0])}' for k, v in tags.items() if v[0] != -1))

        if isinstance(tags, int):
            return [tags]

        if isinstance(tags[0], int):
            return list(set(tags))

        return list(set([x[0] for x in tags]))

    @staticmethod
    def _get_instance_multiplier(tags, class_wise_multipliers, weight_upper, weight_lower):
        mul = numpy.prod([class_wise_multipliers[tag] for tag in tags])

        return BalancedInstanceWeightsGenerator._scope_multiplier(mul, weight_upper, weight_lower)

    @staticmethod
    def _scope_multiplier(value, weight_upper, weight_lower):
        return min(max(value, weight_lower), weight_upper)
