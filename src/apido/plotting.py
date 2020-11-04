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
    plt.figure(
        figsize=(2 * target.shape[-1] * ncols, 2.5 * (target.shape[-1] + 1))
    )
    for col in range(ncols):
        plt.subplot(1 + target.shape[-1], ncols, col + 1)
        plt.imshow(brightfield[col, :, :, 3], vmin=0, vmax=4000)
        plt.axis("off")

        for row in range(target.shape[-1]):
            plt.subplot(4, ncols * 2, ncols * (row + 1) * 2 + 1 + col * 2)
            plt.imshow(prediction[col, :, :, row], vmin=0, vmax=4000)
            plt.axis("off")
            plt.subplot(4, ncols * 2, ncols * (row + 1) * 2 + 2 + col * 2)
            plt.imshow(target[col, :, :, row], vmin=0, vmax=4000)
            plt.axis("off")

        plt.subplots_adjust(hspace=0.02, wspace=0.02)
    return plt
