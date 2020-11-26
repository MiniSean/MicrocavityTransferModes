import numpy as np
from scipy.signal import argrelextrema
from typing import List


class PeakData:
    def __init__(self):
        pass


def identify_peaks(meas_data: np.ndarray):  # -> List[PeakData]:
    # np.r_[True, a[1:] < a[:-1]] & numpy.r_[a[:-1] < a[1:], True]
    return argrelextrema(meas_data, np.less)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.plot_npy import prepare_measurement_plot, plot_class
    # Construct measurement class
    ax, measurement_class = prepare_measurement_plot()
    # Optional, define slice
    slice = (1050000, 1150000)
    measurement_class.slicer = slice  # Zooms in on relevant data part

    # Collect peaks
    peak_indices = identify_peaks(measurement_class.y_data)

    # Apply axis draw/modification
    ax = plot_class(axis=ax, measurement_class=measurement_class)

    # Draw peak v-lines
    # for i, index in enumerate(peak_indices[0]):
    #     if i > 100:
    #         break
    #     ax.axvline(x=index)

    # Show
    plt.show()
