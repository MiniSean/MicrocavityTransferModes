import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Tuple
from src.import_data import import_npy, slice_array, SyncMeasData


def plot_class(axis: plt.axes, measurement_class: SyncMeasData):
    # Plot array
    axis.plot(measurement_class.x_data, measurement_class.y_data)
    # Set axis
    axis.set_xlabel('Sampling Voltage [V]')
    axis.set_ylabel('Transmission [a.u.]')
    axis.set_yscale('log')
    axis.grid(True)
    return axis


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


def prepare_measurement_plot() -> Tuple[plt.axes, SyncMeasData]:
    # Allows to surpass the hardcoded limit in the number of points in the backend Agg
    rcParams['agg.path.chunksize'] = 1000
    # Define file name to retrieve from predefined data path
    file_meas = 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration0'
    file_samp = 'samples_1s_10V_rate1300000.0'
    # Store plot figure and axis
    _, result_axis = plt.subplots()
    # Construct measurement class
    result_class = SyncMeasData(meas_file=file_meas, samp_file=file_samp, scan_file=None)
    return result_axis, result_class


if __name__ == '__main__':
    # Allows to surpass the hardcoded limit in the number of points in the backend Agg
    rcParams['agg.path.chunksize'] = 1000
    # Define file name to retrieve from predefined data path
    file_meas = 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration0'
    file_samp = 'samples_1s_10V_rate1300000.0'
    # Store plot figure and axis
    fig, ax = plt.subplots()
    # Optional, define slice
    slice = (1050000, 1150000)
    # Construct measurement class
    measurement_class = SyncMeasData(meas_file=file_meas, samp_file=file_samp, scan_file=None)

    # Apply axis draw/modification
    # ax = plot_npy(axis=ax, measurement_file=file_meas, sample_file=file_samp, slice=None)
    ax = plot_class(axis=ax, measurement_class=measurement_class)

    fig2, ax2 = plt.subplots()
    measurement_class.slicer = slice
    # ax2 = plot_npy(axis=ax2, measurement_file=file_meas, sample_file=file_samp, slice=slice)
    ax2 = plot_class(axis=ax2, measurement_class=measurement_class)
    # Show figure plot
    plt.show()
