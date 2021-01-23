import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Tuple, Union, List
from src.import_data import import_npy, slice_array, SyncMeasData, FileToMeasData
from src.peak_identifier import PeakCollection, PeakData
from src.peak_relation import LabeledPeakCollection, LabeledPeakCluster, flatten_clusters
from src.peak_normalization import NormalizedPeakCollection, NormalizedPeak
# Allows to surpass the hardcoded limit in the number of points in the backend Agg
rcParams['agg.path.chunksize'] = 1000


def plot_class(axis: plt.axes, measurement_class: SyncMeasData, **kwargs):
    # Plot array
    axis.plot(measurement_class.x_data, measurement_class.y_data, **kwargs)
    return get_standard_axis(axis=axis)


def get_standard_axis(axis: plt.axes) -> plt.axes:
    # Set axis
    axis.set_xlabel('Sampling Cavity Length [nm]')  # Voltage [V]
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
    for cluster in (data if isinstance(data, list) else data.get_clusters):
        if cluster.get_transverse_mode_id == 0:
            plt.gca().set_prop_cycle(None)
        axis = plot_peak_collection(axis=axis, data=cluster, label=f'n+m={cluster.get_transverse_mode_id}')
        axis.text(x=cluster.get_avg_x, y=cluster.get_max_y, s=f'({cluster.get_longitudinal_mode_id}, {cluster.get_transverse_mode_id})', fontsize=8, horizontalalignment='center', verticalalignment='bottom')
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
    result_class = FileToMeasData(meas_file=file_meas, samp_file=file_samp)
    return result_axis, result_class


def plot_peak_identification(collection: PeakCollection, meas_class: SyncMeasData) -> plt.axes:
    # Store plot figure and axis
    _, _ax = plt.subplots()
    _ax = plot_class(axis=_ax, measurement_class=meas_class)
    for i, peak_data in enumerate(collection):
        if peak_data.relevant:
            _ax.plot(peak_data.get_x, peak_data.get_y, 'x', color='r', alpha=1)
    return _ax


def plot_peak_relation(collection: LabeledPeakCollection, meas_class: SyncMeasData) -> plt.axes:
    # Store plot figure and axis
    _, _ax = plt.subplots()
    _ax = plot_class(axis=_ax, measurement_class=meas_class)

    # Determine mode sequence corresponding to first FSR
    cluster_array, value_slice = collection.get_mode_sequence(long_mode=0)
    _ax.axvline(x=value_slice[0], color='r', alpha=1)
    _ax.axvline(x=value_slice[1], color='g', alpha=1)
    _ax = plot_cluster_collection(axis=_ax, data=collection)
    return _ax


def plot_peak_normalization_spectrum(collection: NormalizedPeakCollection) -> plt.axes:
    # Store plot figure and axis
    def peak_inclusion(peak: NormalizedPeak) -> bool:
        return peak.get_norm_x is not None and 0 <= peak.get_norm_x <= 1

    _, _ax = plt.subplots()
    _ax = get_standard_axis(axis=_ax)
    for i in range(max(collection.q_dict.keys())):
        try:
            cluster_array, value_slice = collection.get_mode_sequence(long_mode=i)
            # Get normalized measurement
            x_sample, y_measure = collection.get_measurement_data_slice(union_slice=value_slice)
            _ax.plot(x_sample, y_measure, alpha=1)
            # Get normalized peaks
            peak_array = flatten_clusters(data=cluster_array)
            y = [peak.get_y for peak in peak_array if peak_inclusion(peak)]
            x = [peak.get_x for peak in peak_array if peak_inclusion(peak)]
            _ax.plot(x, y, 'x', alpha=1)
        except AttributeError:
            break
    return _ax


def plot_peak_normalization_overlap(collection: NormalizedPeakCollection) -> plt.axes:
    # Store plot figure and axis
    def peak_inclusion(peak: NormalizedPeak) -> bool:
        return peak.get_norm_x is not None and 0 <= peak.get_norm_x <= 1

    alpha = 0.5
    _, _ax = plt.subplots()
    _ax = get_standard_axis(axis=_ax)
    for i in range(max(collection.q_dict.keys())):
        try:
            cluster_array, value_slice = collection.get_mode_sequence(long_mode=i)
            # Get normalized measurement
            x_sample, y_measure = collection.get_normalized_meas_data_slice(union_slice=value_slice)
            _ax.plot(x_sample, y_measure, alpha=alpha)
            # Get normalized peaks
            peak_array = flatten_clusters(data=cluster_array)
            y = [peak.get_y for peak in peak_array if peak_inclusion(peak)]
            x = [peak.get_norm_x for peak in peak_array if peak_inclusion(peak)]
            _ax.plot(x, y, 'x', alpha=alpha)
        except AttributeError:
            break
    _ax.set_xlabel('Normalized distance ' + r'$[\lambda / 2]$')
    return _ax


if __name__ == '__main__':
    # Define file name to retrieve from predefined data path
    file_meas = 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration0'
    file_samp = 'samples_1s_10V_rate1300000.0'
    # Store plot figure and axis
    fig, ax = plt.subplots()
    # Optional, define data_slice
    slice = (1050000, 1150000)
    # Construct measurement class
    measurement_class = FileToMeasData(meas_file=file_meas, samp_file=file_samp)

    # Apply axis draw/modification
    # ax = plot_npy(axis=ax, measurement_file=file_meas, sample_file=file_samp, data_slice=None)
    ax = plot_class(axis=ax, measurement_class=measurement_class)

    fig2, ax2 = plt.subplots()
    measurement_class.slicer = slice
    # ax2 = plot_npy(axis=ax2, measurement_file=file_meas, sample_file=file_samp, data_slice=data_slice)
    ax2 = plot_class(axis=ax2, measurement_class=measurement_class)
    # Show figure plot
    plt.show()
