import apido
from tensorflow import keras
import os
import bmidt

model = bmidt.get_model(0)[1].generator


_model = apido.load_model(
    os.path.abspath(
        "./results/bmidt/models/loss257bmidt_04-11-2020T040411_model_2"
    ),
    compile=False,
)

model.set_weights(_model.get_weights())
model.summary()
model.compile(loss=apido.combined_metric(), metrics=apido.metrics())

loader = bmidt.get_generator(0)
validation_set = apido.get_validation_set()

prediction = model.evaluate(validation_set[0], validation_set[1], batch_size=2)
