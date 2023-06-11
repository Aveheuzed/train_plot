#!/usr/bin/env python3

import pathlib
import numpy as np
from matplotlib import pyplot as plt


G = 9.81

FILEPATH = pathlib.Path(__file__).parent / "data" / "accel.csv"

# times when the train is known to be static
# tuples of (start_time, end_time)
KNOWN_ZEROES = [
    (0.0, 47.0), # Angers
    (540.0, 570.0), # Savennières
    (775.0, 830.0), # La Possonière
    (1135.0, 1175.0), # Chalonnes
    (1710.0, 1755.0), # Chemillé
    # (), # Cholet, no significant data there
]


def csv_data_formatter(raw):
    """
    Converts a raw item read from csv to a float.
    Quirks:
    - the decimal separator may be a comma (",") or a dot ("."),
    - the value may be nevative (heading "-" sign)
    Takes as input a str, returns a float.
    """
    data = raw.replace(",", ".")
    return float(data)

def load_data(path):
    """
    Loads the CSV file, parses it, and yields the measurements
    as tuples of floats (time, (accel_x, accel_y, accel_z)).
    Returns the data as a 2-dimensional NumPy array.
    """
    with open(path) as file:
        return np.loadtxt(
            file,
            delimiter=";",
            converters={
                0: csv_data_formatter,
                1: csv_data_formatter,
                2: csv_data_formatter,
                3: csv_data_formatter,
            },
            # names=("t", "x..", "y..", "z.."),
            skiprows=1,
            usecols=(0, 1, 2, 3),
            encoding="utf-8",
        )

def normalize(data):
    """
    Nomalizes the data:
    converts from g to m.s^-2,
    changes the orientation to (forward, right, up),

    All this is done **in-place**; the modified input array is returned.
    """

    # reorientation
    data *= (1.0, -1.0, -1.0, 1.0)

    # conversion to S.I.
    data[::, 1:] *= G

    return data

def calibrate(data, static_times):
    """
    Calibrates the data to remove any systematic bias
    (both static error from the sensor, and the Earth's pull).
    This operation is done using the periods of time
    when the system is assumed to be static, i.e. train stops.

    static_times is an iterable of pairs (time_start, time_stop)
    when the system is assumed unmoving.

    This function operates **in-place**: the calibrated input array is returned.
    """

    # 1 when the system is static, 0 otherwise
    mask = np.any(
        np.stack(
            [(start < data[::, 0]) & (data[::, 0] < stop) for start, stop in static_times],
            axis=1,
        ),
        axis=1,
    )

    bias = np.mean(data[mask, 1:], axis=0)

    data[::, 1:] -= bias

    return data

def plot_serie(ax, time, values, legend, ylabel):
    """
    Plots the given value set (1-dim) as a function of time,
    with the relevant title.
    Returns the resulting matplotlib.lines.Line2D object.
    """
    ax.set_ylabel(ylabel)
    ax.grid(which="both", axis="both")

    lines = ax.plot(time, values, label=legend)

    ax.legend(loc="best")

    return lines


if __name__ == "__main__":
    arr = load_data(FILEPATH)
    arr = normalize(arr)
    arr = calibrate(arr, KNOWN_ZEROES)

    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, sharex=True)

    plot_serie(ax_x, arr[::, 0], arr[::, 1], "x..", "x.. (m.s^-2)")
    plot_serie(ax_y, arr[::, 0], arr[::, 2], "y..", "y.. (m.s^-2)")
    plot_serie(ax_z, arr[::, 0], arr[::, 3], "z..", "z.. (m.s^-2)")

    ax_z.set_xlabel("time (s)")

    plt.show()
