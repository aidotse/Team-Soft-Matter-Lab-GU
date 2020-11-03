import matplotlib.pyplot as plt
import os
import apido
import numpy as np


def user_results(username):
    path = os.path.join("./results", username, "csv")
    os.makedirs(path, exist_ok=True)
    paths = os.listdir(path)
    for file in paths:
        if not file.endswith(".csv"):
            continue

        _, result = apido.read_csv(os.path.join(path, file))
        yield (file, result)


def all_results():
    for user in os.listdir("./results"):
        yield (user, user_results(user))


def plot_results(metric="val_loss"):
    users = []
    for user, user_iterator in all_results():
        users.append(user)

        time_values = []
        scores = []
        for filename, result_dict in user_iterator:

            try:
                score = np.min(result_dict[metric])
                date = apido.get_date_from_filename(filename)

                time_values.append(date)
                scores.append(score)

            except KeyError as e:
                # Skip
                print(e)
                pass
        time_values, scores = zip(*sorted(zip(time_values, scores)))

        scores = list(scores)
        for idx in range(1, len(scores)):
            if scores[idx] > scores[idx - 1]:
                scores[idx] = scores[idx - 1]

        plt.plot(time_values, scores, marker="o")

    plt.legend(users)
    plt.xlabel("Date")
    plt.ylabel(metric)

    return plt


def plot_evaluation(brightfield, target, prediction, ncols=5):
    plt.figure(figsize=(15, 5))
    for col in range(ncols):
        plt.subplot(4, ncols, col + 1)
        plt.imshow(brightfield[col, :, :, 3])
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 2 + 1 + col * 2)
        plt.imshow(prediction[col, :, :, 0])
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 2 + 2 + col * 2)
        plt.imshow(target[col, :, :, 0])
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 4 + 1 + col * 2)
        plt.imshow(prediction[col, :, :, 1])
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 4 + 2 + col * 2)
        plt.imshow(target[col, :, :, 1])
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 6 + 1 + col * 2)
        plt.imshow(prediction[col, :, :, 2])
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 6 + 2 + col * 2)
        plt.imshow(target[col, :, :, 2])
        plt.axis("off")
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    return plt