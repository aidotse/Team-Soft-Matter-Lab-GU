import matplotlib.pyplot as plt
import os
import apido
import numpy as np
import glob


def user_results(username):
    folders = glob.glob(
        os.path.abspath(os.path.join("./results", username, "**", "*.csv"))
    )

    for file in folders:
        if not file.endswith(".csv"):
            continue

        result = apido.read_csv(file)
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

        if time_values and scores:
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
    plt.figure(figsize=(6 * ncols, 10))
    for col in range(ncols):
        plt.subplot(4, ncols, col + 1)
        plt.imshow(brightfield[col, :, :, 3], vmin=0, vmax=4000)
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 2 + 1 + col * 2)
        plt.imshow(prediction[col, :, :, 0], vmin=0, vmax=3000)
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 2 + 2 + col * 2)
        plt.imshow(target[col, :, :, 0], vmin=0, vmax=3000)
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 4 + 1 + col * 2)
        plt.imshow(prediction[col, :, :, 1], vmin=0, vmax=5000)
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 4 + 2 + col * 2)
        plt.imshow(target[col, :, :, 1], vmin=0, vmax=5000)
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 6 + 1 + col * 2)
        plt.imshow(prediction[col, :, :, 2], vmin=0, vmax=3000)
        plt.axis("off")
        plt.subplot(4, ncols * 2, ncols * 6 + 2 + col * 2)
        plt.imshow(target[col, :, :, 2], vmin=0, vmax=3000)
        plt.axis("off")
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    return plt