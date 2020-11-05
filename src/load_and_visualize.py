import apido
from tensorflow import keras
import os
import bmidt
from PIL import Image
import numpy as np
import sys

specifier = sys.argv[1]

file_struct = "AssayPlate_Greiner_#655090_{1}_T0001F{2}L01A0{3}Z01C0{3}.tif"

target_folder = apido.get_folder_from_specifier(specifier)

print("Creating predictions for", target_folder)

model = apido.load_model(target_folder)

loader = bmidt.get_generator(0)
validation_set = apido.get_validation_set(convert_to_array=False)

prediction = model.predict(np.array(validation_set[0]), batch_size=1)

prediction[prediction < 0] = 0

os.makedirs(os.path.join(target_folder, "predictions"), exist_ok=True)
os.makedirs(os.path.join(target_folder, "targets"), exist_ok=True)


for pred, target in zip(prediction, validation_set[1]):
    well = target.get_property("index_well")
    site = target.get_property("index_site")

    for action in range(3):
        file_path = file_struct.format(well, site, action + 1, 1)

        p_layer = Image.fromarray(pred[..., action].astype(np.uint16))
        t_layer = Image.fromarray(target[..., action].astype(np.uint16))

        p_layer.save(os.path.join(target_folder, "predictions", file_path))
        p_layer.save(os.path.join(target_folder, "targets", file_path))

print("Done!")