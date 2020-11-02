import re
import csv
import getpass
import datetime
import numpy as np
import itertools

_checkpoint_struct = "{0}_{1}_model_{2}"
_datestring_struct = "%d-%m-%YT%H;%M;%S"


def save_history_as_csv(
    path: str, history: dict, headers: dict, delimiter="\t"
):
    """Saves the result of a keras training session as a csv.

    The output format is
    .. codeblock:

       loss{delim}val_loss{delim}...
       0.1{delim}0.11{delim}...
       0.09{delim}0.1{delim}...

    Parameters
    ----------
    path : str
        Path to save the file to.
    history : dict
        A history object created by a keras training session.
    headers : dict
        Extra heading at the top of the csv
    delimiter : str
        Delimination character placed between entries. Defaults to tab.

    """

    assert path.endswith(".csv"), "File format needs to be .csv"

    num_rows = np.inf

    for v in history.values():
        num_rows = np.min([num_rows, len(v)])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)

        # Write headers
        writer.writerow(headers.keys())
        writer.writerow(headers.values())

        writer.writerow(history.keys())

        # Write rows
        for idx in range(int(num_rows)):
            writer.writerow([v[idx] for v in history.values()])


def read_csv(path, delimiter="\t"):
    result_dict = {}
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)

        # Write headers
        header_names = next(reader)
        header_vals = next(reader)

        headers = dict(zip(header_names, header_vals))

        keys = next(reader)
        for key in keys:
            result_dict[key] = []

        # Write rows
        for row in reader:
            for key, value in zip(keys, row):
                result_dict[key].append(float(value))

    return headers, result_dict


def get_datestring(dtime=None):
    """Return a formated string displaying the current date."""

    if dtime is None:
        dtime = datetime.datetime.now()
    return dtime.strftime(_datestring_struct)


def get_date_from_filename(filename: str):
    """Extract datetime object from filename"""
    datestring = filename.split("_")[1]

    date = datetime.datetime.strptime(datestring, _datestring_struct)
    return date


def get_checkpoint_name(index: int or str, name: str = getpass.getuser()):
    """Return a filename corresponding to a training configuration.

    Paramterers
    -----------
    index : int or str
        Index used for
    """

    return _checkpoint_struct.format(name, get_datestring(), index)


_RE_PATTERN = r"([0-9]*):([0-9]*)(?::([0-9]*)){0,1}"


def parse_index(index: str):
    try:
        return iter([int(index)])
    except ValueError:
        match = re.match(_RE_PATTERN, index)

        if match is None:
            raise ValueError(
                "Could not parse index. Valid examples include: 0, 4:8, :6, 4:, 4:16:4 etc."
            )
        start = int(match.group(1) or 0)
        stop = int(match.group(2) or -1)
        step = int(match.group(3) or 1)

        if stop == -1:
            return itertools.count(start, step)
        else:

            return range(start, stop, step)
