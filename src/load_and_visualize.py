import apido
from tensorflow import keras
import os
import numpy as np
import bmidt

model = bmidt.get_model(0)[1].generator
model.load_weights(
    os.path.abspath(
        "./results/bmidt/models/loss114.87405395507812;root_03-11-2020T16;52;18_model_0.h5"
    ),
)

loader = bmidt.get_generator(0)

validation_set = apido.get_validation_set()

prediction = model.predict(validation_set[0])

np.save(
    "loss114.87405395507812;root_03-11-2020T16;52;18_model_0",
    [prediction, validation_set[1]],
)
