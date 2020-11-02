import numpy as np
from typing import List
import deeptrack as dt
import apido

_VALIDATION_SET_SIZE = 8

conf = {}


def batch_function(image: dt.Image or List[dt.Image]) -> dt.Image:
    return image[0]


def label_function(image: dt.Image or List[dt.Image]) -> dt.Image:
    return image[1]


def get_generator(
    min_data_size=1000,
    max_data_size=2000,
    augmentation_dict={},
    seed=None,
    root_path=".",
    **kwargs
):
    conf["root_path"] = root_path
    args = {
        "feature": apido.DataLoader(
            # augmentation=augmentation_dict,
            seed=seed,
            path=root_path,
        ),
        "label_function": label_function,
        "batch_function": batch_function,
        "min_data_size": min_data_size,
        "max_data_size": max_data_size,
        **kwargs,
    }
    return dt.utils.safe_call(dt.generators.ContinuousGenerator, **args)


def get_validation_set(size=_VALIDATION_SET_SIZE):
    data_loader = apido.DataLoader(path=conf["root_path"])

    data = []
    labels = []
    for _ in range(size):
        data_loader.update(validation=True, is_validation=True)
        output = data_loader.resolve()
        data.append(batch_function(output))
        labels.append(label_function(output))

    return np.array(data), np.array(labels)
