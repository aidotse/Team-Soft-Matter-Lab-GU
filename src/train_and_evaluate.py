# Natives
from os import error
import tensorflow.keras as keras
import tensorflow as tf
import sys
import os
import getopt
import importlib

# Packages
import numpy as np

# Locals
import apido

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Grab passed arguments
opts, args = getopt.getopt(sys.argv[2:], "i:e:p:n:")

script = sys.argv[1]
# Defaults
args = {
    "epochs": 100000,
    "patience": 100,
}

username = apido.get_user_name()

index = None
for opt, arg in opts:
    if opt == "-i":
        index = arg
    elif opt == "-e":
        args["epochs"] = int(arg)
    elif opt == "-p":
        args["patience"] = int(arg)
    elif opt == "-n":
        username = arg

if index is None:
    error("option -i not set")
    sys.exit(0)

indices = apido.parse_index(index)


# Create file structure
PATH_TO_CHECKPOINTS = os.path.abspath(
    os.path.join("results", username, "checkpoints")
)
PATH_TO_MODELS = os.path.abspath(os.path.join("results", username, "models"))
PATH_TO_CSV = os.path.abspath(os.path.join("results", username, "csv"))
PATH_TO_IMAGES = os.path.abspath(os.path.join("results", username, "images"))
PATH_TO_PREDICTIONS = os.path.abspath(
    os.path.join("results", username, "predictions")
)

os.makedirs(PATH_TO_IMAGES, exist_ok=True)
os.makedirs(PATH_TO_PREDICTIONS, exist_ok=True)
os.makedirs(PATH_TO_CHECKPOINTS, exist_ok=True)
os.makedirs(PATH_TO_MODELS, exist_ok=True)
os.makedirs(PATH_TO_CSV, exist_ok=True)


user_models = importlib.import_module(script)

for index in indices:
    try:
        m_header_dict, model = user_models.get_model(index)
        d_header_dict, generator = user_models.get_generator(index)
    except KeyError as e:
        print(e)
        print(
            "The obove error likely occured because the index range was"
            " larger than the number of defined models. If so, you can"
            " safely disregard the error."
        )
        break

    headers = {**m_header_dict, **d_header_dict, **args}

    model.compile(loss=apido.combined_metric(), metrics=apido.metrics())

    print("")

    print("=" * 50, "START", "=" * 50)

    print(
        "Running trial on model {0}, with patience of {1}".format(
            index, args["patience"]
        )
    )

    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args["patience"],
        restore_best_weights=True,
    )

    checkpoint_name = apido.get_checkpoint_name(index, name=username)
    checkpointing = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(PATH_TO_CHECKPOINTS, checkpoint_name),
        save_best_only=True,
    )

    # Define data generators
    print("Grabbing validation data...")
    validation_data = apido.get_validation_set()

    # Start training
    print("Starting training. Model is saved to: {0}".format(checkpoint_name))

    with generator:
        h = model.fit(
            generator,
            epochs=args["epochs"],
            callbacks=[early_stopping],
            validation_data=validation_data,
            validation_batch_size=2,
        )

    predictions = model.predict(validation_data[0][:2], batch_size=2)

    plot = apido.plot_evaluation(
        validation_data[0], validation_data[1], predictions, ncols=2
    )

    plot.savefig(
        os.path.join(PATH_TO_IMAGES, checkpoint_name + ".png"), dpi=600
    )

    # Prediction
    np.save(
        os.path.join(PATH_TO_PREDICTIONS, checkpoint_name),
        [predictions[:2], validation_data[1][:2]],
    )

    # Log results
    results = h.history
    apido.save_history_as_csv(
        os.path.join(PATH_TO_CSV, checkpoint_name) + ".csv",
        results,
        headers=headers,
    )

    best_loss = np.min(results["val_loss"])

    result_path = os.path.join(
        PATH_TO_MODELS, "loss{0};{1}".format(best_loss, checkpoint_name)
    )

    print(
        "Saving best generator as {0}, best loss was {1}".format(
            result_path, best_loss
        )
    )
    model.generator.save(result_path)
