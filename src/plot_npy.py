import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Tuple
DATA_DIR = '../data/Trans'


def import_npy(filename: str) -> np.ndarray:
    filepath = os.path.join(DATA_DIR, filename + '.npy')
    return np.load(file=filepath, allow_pickle=False)


def slice_array(array: np.ndarray, slice: Tuple[int, int]) -> np.ndarray:
    min_index = min(slice)
    max_index = max(slice)
    return array[min_index: max_index]


def plot_npy(axis: plt.axes, measurement_file: str, sample_file: str, slice: Tuple[int, int] = None) -> plt.axes:
    # Import measurement values
    data_array = import_npy(measurement_file)
    reflection_array = data_array[0]
    # Import sample tokens
    samp_array = import_npy(sample_file)

    # Slice if necessary
    if slice is not None:
        reflection_array = slice_array(array=reflection_array, slice=slice)
        samp_array = slice_array(array=samp_array, slice=slice)
    # Plot array
    axis.plot(samp_array, reflection_array)
    # Set axis
    axis.set_xlabel('Sampling Voltage [V]')
    axis.set_ylabel('Transmission [a.u.]')
    axis.set_yscale('log')
    axis.grid(True)
    return axis


if __name__ == '__main__':
    # Allows to surpass the hardcoded limit in the number of points in the backend Agg
    rcParams['agg.path.chunksize'] = 1000
    # Define file name to retrieve from predefined data path
    file_meas = 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration1'
    file_samp = 'samples_1s_10V_rate1300000.0'
    # Store plot figure and axis
    fig, ax = plt.subplots()
    # Optional, define slice
    slice = (1050000, 1150000)
    # Apply axis draw/modification
    ax = plot_npy(axis=ax, measurement_file=file_meas, sample_file=file_samp, slice=None)

    fig2, ax2 = plt.subplots()
    ax2 = plot_npy(axis=ax2, measurement_file=file_meas, sample_file=file_samp, slice=slice)
    # Show figure plot
    plt.show()
