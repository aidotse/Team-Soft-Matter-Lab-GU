import deeptrack as dt
import numpy as np
import itertools
import glob
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


def DataLoader(path=None, magnification="60x", format=".tif", **kwargs):

    # Define path to the dataset
    path_to_dataset = path + magnification + " images/"

    input_root = path_to_dataset + "input/"
    label_root = path_to_dataset + "targets/"

    # Number of input files
    input_files = glob.glob(input_root + "*")

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
    wells_iterator = itertools.cycle(iter(wells))
    site_iterator = itertools.cycle(
        iter(
            range(
                1, props_per_magnification[magnification]["sites_per_well"] + 1
            )
        )
    )

    load_string_struct = (
        "{0}AssayPlate_Greiner_#655090_{1}_T0001F{2}L01A0{3}Z0{4}C0{3}"
        + format
    )

    root = dt.DummyFeature(
        index_well=lambda: next(wells_iterator),
        index_site=lambda: ("00" + str(next(site_iterator)))[-3:],
        # index_site="012",
        index_action_list_number=list(range(1, 4)),
        index_z_slide=list(
            range(1, props_per_magnification[magnification]["z_slides"] + 1)
        ),
    )

    bf = root + dt.LoadImage(
        path=lambda index_well, index_site, index_z_slide: [
            load_string_struct.format(
                input_root, index_well, index_site, 4, z_slide
            )
            for z_slide in index_z_slide
        ],
        **root.properties,
    )
    fl = root + dt.LoadImage(
        path=lambda index_well, index_site, index_action_list_number: [
            load_string_struct.format(
                label_root, index_well, index_site, action_number, 1
            )
            for action_number in index_action_list_number
        ],
        **root.properties,
    )

    dataset = dt.Combine([bf, fl]) + dt.Crop(
        crop=(256, 256, 7),
        corner=lambda: (*np.random.randint(0, 10000, size=2), 0),
    )

    augmented_dataset = Augmentation(dataset)

    return dt.ConditionalSetFeature(
        on_true=dataset,
        on_false=augmented_dataset,
        condition="is_validation",
        is_validation=lambda validation: validation,
    )
