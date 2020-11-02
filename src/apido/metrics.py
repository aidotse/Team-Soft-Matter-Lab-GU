# All metrics

from tensorflow.keras import backend as K


def mae(P, T):
    return K.mean(K.abs(P - T))


_metrics = [mae]


def metrics():
    """Returns a list of all metrics.

    Returns
    -------
    list
        List of metric functions
    """
    return list(_metrics)


def combined_metric(weights=[1, 1, 1, 1]):
    """Weighted sum of metrics

    Parameters
    ----------
    weights : list, optional
        A list of weights for the individual metrics. Should be the same
        length as returned by `metrics`.

    Returns
    -------
    Callable[Tensor, Tensor] -> Tensor
        A tensorflow/keras metrics function.
    """

    def inner(P, T):
        return sum(
            weight * metric(P, T) for weight, metric in zip(weights, metrics())
        )

    return inner
