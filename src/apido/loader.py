import deeptrack as dt
import numpy as np
import itertools
import glob
import random
import re

props_per_magnification = {
    "20x": {"wells": 9, "sites_per_well": 6, "z_slides": 7},
    "40x": {"wells": 9, "sites_per_well": 8, "z_slides": 7},
    "60x": {"wells": 9, "sites_per_well": 12, "z_slides": 7},
}


default_augmentation_list = {
    "FlipLR": {},
    "FlipUD": {},
    "FlipDiagonal": {},
    "Affine": {
        "rotate": lambda: np.random.rand() * 2 * np.pi,
        "shear": lambda: np.random.rand() * 0.3 - 0.15,
        "scale": {
            "x": np.random.rand() * 0.3 + 0.85,
            "y": np.random.rand() * 0.3 + 0.85,
        },
        "mode": "reflect",
    },
}


def Augmentation(
    image: dt.Feature,
    augmentation_list=default_augmentation_list.copy(),
    default_value=lambda x: x,
    **kwargs
):

    augmented_image = image
    for augmentation in augmentation_list:
        augmented_image = (
            getattr(dt, augmentation, default_value)(
                **augmentation_list.get(augmentation), **kwargs
            )
            + augmented_image
        )

    return augmented_image


def DataLoader(
    path=None,
    magnification="60x",
    format=".tif",
    augmentation=None,
    training_split=0.7,
    seed=None,
    **kwargs
):

    # Define path to the dataset
    path_to_dataset = path + magnification + " images/"
    print(path_to_dataset)
    # Number of input files
    input_files = glob.glob(path_to_dataset + "*.tif")

    # Compute well coordinates
    wells = list(
        dict.fromkeys(
            [
                re.findall(r"(?<=AssayPlate_Greiner_#655090_).*?(?=_)", file)[
                    0
                ]
                for file in input_files
            ]
        )
    )

    # Iterate over wells and sites

    site_config = list(
        itertools.product(
            wells,
            range(
                1, props_per_magnification[magnification]["sites_per_well"] + 1
            ),
        )
    )

    split = int((len(site_config) * training_split))

    if seed:
        random.seed(seed)
    random.shuffle(site_config)

    training_iterator = itertools.cycle(site_config[:split])
    validation_iterator = itertools.cycle(site_config[split:])

    load_string_struct = (
        "{0}AssayPlate_Greiner_#655090_{1}_T0001F{2}L01A0{3}Z0{4}C0{3}"
        + format
    )

    root = dt.DummyFeature(
        well_site_tuple=lambda is_validation: next(validation_iterator)
        if is_validation
        else next(training_iterator),
        index_well=lambda well_site_tuple: well_site_tuple[0],
        index_site=lambda well_site_tuple: ("00" + str(well_site_tuple[1]))[
            -3:
        ],
        index_action_list_number=[1, 2, 3],
        index_z_slide=list(
            range(1, props_per_magnification[magnification]["z_slides"] + 1)
        ),
    )

    bf = root + dt.LoadImage(
        path=lambda index_well, index_site, index_z_slide: [
            load_string_struct.format(
                path_to_dataset, index_well, index_site, 4, z_slide
            )
            for z_slide in index_z_slide
        ],
        **root.properties,
    )
    fl = root + dt.LoadImage(
        path=lambda index_well, index_site, index_action_list_number: [
            load_string_struct.format(
                path_to_dataset, index_well, index_site, action_number, 1
            )
            for action_number in index_action_list_number
        ],
        **root.properties,
    )

    dataset = dt.Combine([bf, fl]) + dt.CropToMultiplesOf(multiple=32)

    if augmentation:
        augmented_dataset = Augmentation(dataset)
    else:
        augmented_dataset = dataset

    augmented_dataset = dt.PreLoad(dataset, updates_per_reload=16)

    augmented_dataset += dt.Crop(
        crop=(256, 256, 7),
        corner=lambda: (*np.random.randint(0, 10000, size=2), 0),
    )

    return dt.ConditionalSetFeature(
        on_true=dataset,
        on_false=augmented_dataset,
        condition="is_validation",
        is_validation=lambda validation: validation,
    )
