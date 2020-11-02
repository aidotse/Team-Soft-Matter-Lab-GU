import numpy as np
from typing import List
import deeptrack as dt

_VALIDATION_SET_SIZE = 8


def feature() -> dt.Feature:
    fluorescence = dt.Fluorescence(
        magnification=5, output_region=(0, 0, 256, 256)
    )

    particles = dt.Ellipsoid(
        radius=lambda: np.random.rand(3) * 0.5e-6 + 0.5e-6,
        rotation=lambda: np.random.rand(3) * 2 * np.pi,
        position=lambda: np.random.rand(2) * 256,
        z=lambda: np.random.randn() * 15,
    )

    image_feature = fluorescence(
        particles ** (lambda: np.random.randint(5, 20))
    )

    label_feature = dt.Bind(image_feature, z=0)

    pipeline = dt.Combine([image_feature, label_feature])
    pipeline += dt.NormalizeMinMax(-1, 1)

    return pipeline


def batch_function(image: dt.Image or List[dt.Image]) -> dt.Image:
    return image[0]


def label_function(image: dt.Image or List[dt.Image]) -> dt.Image:
    return image[1]


def get_generator(min_data_size=1000, max_data_size=2000, **kwargs):

    args = {
        "feature": feature(),
        "label_function": label_function,
        "min_data_size": min_data_size,
        "max_data_size": max_data_size,
        **kwargs,
    }
    return dt.utils.safe_call(dt.generators.ContinuousGenerator, **args)


def get_validation_set(size=_VALIDATION_SET_SIZE):
    data_loader = feature()

    data = []
    labels = []
    for _ in range(size):
        data_loader.update(is_valdidation=True)
        output = data_loader.resolve()
        data.append(batch_function(output))
        labels.append(label_function(output))

    return np.array(data), np.array(labels)
