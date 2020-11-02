# Natives
import tensorflow.keras as keras
import sys
import os
import getopt
import importlib

# Packages
import getpass
import numpy as np

# Locals
import apido

# Grab passed arguments
opts, args = getopt.getopt(sys.argv[1:], "i:e:p:b:n:")
print(opts, args)
# Defaults
args = {
    "epochs": 100,
    "patience": 10,
    "batch_size": 16,
}

username = getpass.getuser()

for opt, arg in opts:
    if opt == "-i":
        index = arg
    elif opt == "-e":
        args["epochs"] = arg
    elif opt == "-p":
        args["patience"] = arg
    elif opt == "-b":
        args["batch_size"] = arg
    elif opt == "-n":
        username = arg


# User import
user_models = importlib.import_module(username)

m_header_dict, model = user_models.get_model(index)
d_header_dict, generator = user_models.get_generator(index)

headers = {**m_header_dict, **d_header_dict, **args}

# Create file structure
PATH_TO_CHECKPOINTS = os.path.join("results", username, "checkpoints")
PATH_TO_MODELS = os.path.join("results", username, "models")
PATH_TO_CSV = os.path.join("results", username, "csv")

os.makedirs(PATH_TO_CHECKPOINTS, exist_ok=True)
os.makedirs(PATH_TO_MODELS, exist_ok=True)
os.makedirs(PATH_TO_CSV, exist_ok=True)


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
    monitor="val_loss", patience=args["patience"], restore_best_weights=True
)

checkpoint_name = apido.get_checkpoint_name(index, args["batch_size"])
checkpointing = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(PATH_TO_CHECKPOINTS, checkpoint_name),
    save_best_only=True,
)

# Define data generators
print("Grabbing validation data...")
validation_data = apido.get_validation_set()

# Start training
print("Starting training. Model is saved to: {0}.h5".format(checkpoint_name))
with generator:
    h = model.fit(
        generator,
        epochs=args["epochs"],
        callbacks=[early_stopping, checkpointing],
        validation_data=validation_data,
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
    PATH_TO_MODELS, "loss{0};{1}.h5".format(best_loss, checkpoint_name)
)

print(
    "Saving best generator as {0}, best loss was {1}".format(
        result_path, best_loss
    )
)
model.generator.save(result_path)
