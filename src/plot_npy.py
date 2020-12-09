import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Tuple, Union, List
from src.import_data import import_npy, slice_array, SyncMeasData
from src.peak_identifier import PeakCollection, PeakData
from src.peak_relation import LabeledPeakCollection, LabeledPeak, LabeledPeakCluster


def plot_class(axis: plt.axes, measurement_class: SyncMeasData):
    # Plot array
    axis.plot(measurement_class.x_data, measurement_class.y_data)
    return get_standard_axis(axis=axis)


def get_standard_axis(axis: plt.axes) -> plt.axes:
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
    return get_standard_axis(axis=axis)


def plot_peak_collection(axis: plt.axes, data: Union[List[PeakData], PeakCollection], label: str = '') -> plt.axes:
    # Plot peaks
    x = [peak.get_x for peak in data]
    y = [peak.get_y for peak in data]
    axis.plot(x, y, 'x', label=label)
    return get_standard_axis(axis=axis)


def plot_cluster_collection(axis: plt.axes, data: Union[List[LabeledPeakCluster], LabeledPeakCollection]) -> plt.axes:
    for cluster in (data.get_clusters if isinstance(data, LabeledPeakCollection) else data):
        axis = plot_peak_collection(axis=axis, data=cluster, label=f'n+m={cluster.get_transverse_mode_id}')
    return axis


def plot_specific_peaks(axis: plt.axes, data: LabeledPeakCollection, long_mode: Union[int, None], trans_mode: Union[int, None]) -> plt.axes:
    filtered_data = data.get_labeled_peaks(long_mode=long_mode, trans_mode=trans_mode)
    return plot_peak_collection(axis=axis, data=filtered_data, label=f'Mode(L:{"all" if long_mode is None else long_mode}, T:{"all" if trans_mode is None else trans_mode})')


def prepare_measurement_plot(measure_file: str = 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration0') -> Tuple[plt.axes, SyncMeasData]:
    # Allows to surpass the hardcoded limit in the number of points in the backend Agg
    rcParams['agg.path.chunksize'] = 1000
    # Define file name to retrieve from predefined data path
    file_meas = measure_file
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
    # Optional, define data_slice
    slice = (1050000, 1150000)
    # Construct measurement class
    measurement_class = SyncMeasData(meas_file=file_meas, samp_file=file_samp, scan_file=None)

    # Apply axis draw/modification
    # ax = plot_npy(axis=ax, measurement_file=file_meas, sample_file=file_samp, data_slice=None)
    ax = plot_class(axis=ax, measurement_class=measurement_class)

    fig2, ax2 = plt.subplots()
    measurement_class.slicer = slice
    # ax2 = plot_npy(axis=ax2, measurement_file=file_meas, sample_file=file_samp, data_slice=data_slice)
    ax2 = plot_class(axis=ax2, measurement_class=measurement_class)
    # Show figure plot
    plt.show()
