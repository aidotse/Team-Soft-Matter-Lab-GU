import glob
import getopt
import os
import sys
import apido
import numpy as np
import matplotlib.pyplot as plt

args = sys.argv[2:]
a, opts = getopt.getopt(args, "be:n:")

metric = sys.argv[1]

name = "*"
epoch_limit = 0
best = False

for opt, val in a:
    if opt == "-b":
        best = True
    if opt == "-e":
        epoch_limit = int(val)
    if opt == "-n":
        name = val

path = os.path.join("./results", name, "loss=*")

folders = glob.glob("./results/*/loss=*/")

at_hundred = []
at_thousand = []
at_best = []

for folder in folders:
    try:
        config = apido.load_config(folder)

        csv = apido.read_csv(os.path.join(folder, "training_history.csv"))

        results = csv["val_loss"]

        config_value = config[metric]

        if len(results) > epoch_limit:
            at_best.append((config_value, np.min(results)))
        if len(results) > 110:
            at_hundred.append((config_value, np.mean(results[90:110])))
        if len(results) > 1020:
            at_thousand.append((config_value, np.mean(results[980:1020])))
    except Exception as e:
        print(e)

print(*zip(*at_best))

plt.subplot(1, 3, 1)
try:
    plt.scatter(*zip(*at_hundred))
except:
    pass

plt.subplot(1, 3, 2)
try:
    plt.scatter(*zip(*at_thousand))
except:
    pass

plt.subplot(1, 3, 3)
try:
    plt.scatter(*zip(*at_best))
except:
    pass

plt.show()