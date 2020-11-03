import apido
from tensorflow import keras
import os
import bmidt

model = keras.models.load_model(
    os.path.abspath(
        "./results/bmidt/models/loss0.29744723439216614;bmidt_03-11-2020T15;15;20_model_0.h5"
    ),
    compile=False,
)

loader = bmidt.get_generator(0)
validation_set = apido.get_validation_set()

prediction = model.predict(validation_set[0])

plt = apido.plot_evaluation(validation_set[0], validation_set[1], prediction)
plt.show()
