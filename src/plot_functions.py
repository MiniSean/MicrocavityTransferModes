import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Tuple, Union, List
from src.import_data import import_npy, slice_array, SyncMeasData, FileToMeasData
from src.peak_identifier import PeakCollection, PeakData
from src.peak_relation import LabeledPeakCollection, LabeledPeakCluster, flatten_clusters
from src.peak_normalization import NormalizedPeakCollection, NormalizedPeak
from src.cavity_radius_analysis import get_gouy_func
# Allows to surpass the hardcoded limit in the number of points in the backend Agg
rcParams['agg.path.chunksize'] = 1000


def plot_class(axis: plt.axes, measurement_class: SyncMeasData, **kwargs):
    # Plot array
    axis.plot(measurement_class.x_data, measurement_class.y_data, **kwargs)
    return get_standard_axis(axis=axis)


def get_standard_axis(axis: plt.axes) -> plt.axes:
    # Set axis
    axis.set_xlabel('Sampling Cavity Length [nm]')  # Voltage [V]
    axis.set_ylabel('Transmission (log10(V))')
    axis.set_yscale('log')
    locs, labels = plt.yticks()
    yticks = [np.log10(value) for value in locs[0:-1]]
    plt.yticks(locs[0:-1], yticks)
    axis.set_ylim([10**(-4), 10**(1)])  # Custom y-lim
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
    try:
        cluster_array, value_slice = collection.get_mode_sequence(long_mode=collection.get_min_q_id)
        _ax.axvline(x=value_slice[0], color='r', alpha=1)
        _ax.axvline(x=value_slice[1], color='g', alpha=1)
    except ValueError:
        pass
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


def plot_radius_estimate(collection: LabeledPeakCollection, radius_mean: float, offset: float, radius_std: float) -> plt.axes:
    _, _ax = plt.subplots()
    cluster_array = collection.get_clusters
    cluster_array.extend([cluster for cluster in collection.get_q_clusters])
    for i in range(-1, 6):
        sub_cluster_array = [cluster for cluster in cluster_array if cluster.get_transverse_mode_id == i]
        if len(sub_cluster_array) == 0:
            continue

        d = np.asarray([cluster.get_avg_x for cluster in sub_cluster_array])
        # Plot Gouy prediction
        y_array = get_gouy_func(length_input=d, length_offset=offset, radius_mean=radius_mean, trans_mode=i)
        _ax.plot(d, y_array, 'o', label=f'm+n={i}')
        # Plot fit lines
        x_space = np.linspace(-offset, d[-1], 100)
        y_space = get_gouy_func(length_input=x_space, length_offset=offset, radius_mean=radius_mean, trans_mode=i)
        _ax.plot(x_space, y_space, ('k:' if i == -1 else 'k-'))

    # Vertical indicator lines
    _ax.axvline(x=-offset, ls='--', color='darkorange')
    _ax.axvline(x=0, ls='--', color='darkorange')

    # Produce fit results
    print(f'Cavity radius: R = {round(radius_mean, 1)} ' + r'+/-' + f' {round(radius_std, 1)} [nm]')
    print(f'Cavity offset length: {offset} [nm]')
    _ax.set_title(f'Fitted cavity radius: R={round(radius_mean, 1)} ' + r'$\pm$' + f' {round(radius_std, 1)} [nm]')
    _ax.set_ylabel(f'Transverse mode splitting' + r' [$\Delta L / (\lambda / 2)$]')
    _ax.set_xlabel(f'Mirror position - {round(offset, 1)} (Offset)' + r' [nm]')
    _ax.grid(True)
    _ax.legend()
    return _ax


# Define font
font_size = 22
plt.rcParams.update({'font.size': font_size})


def plot_allan_variance(xs: np.ndarray, ys: np.ndarray) -> plt.axes:
    _, _ax = plt.subplots()
    _ax.plot(xs, ys, '.')
    _ax.set_ylabel(f'Variance' + r' $\langle [x(t + \delta t) - x(t)]^2 \rangle$ $[nm^2]$', fontsize=font_size)
    # _ax.set_ylabel(f'Distance' + r' $abs(x(t + \delta t) - x(t))$ $[nm]$', fontsize=font_size)
    _ax.set_xlabel(f'Time step' + r' $\delta t[s]$', fontsize=font_size)
    _ax.set_yscale('log')
    # _ax.set_xscale('log')
    _ax.grid(True)
    return _ax


def plot_mode_classification(meas_data: SyncMeasData) -> plt.axes:
    """Plots report paper figure for entire classification process"""
    from src.peak_identifier import identify_peaks
    from src.peak_relation import LabeledPeakCollection, get_converted_measurement_data
    from src.main import Q_OFFSET
    _fig, ((_ax00, _ax01), (_ax10, _ax11)) = plt.subplots(2, 2, sharey='all')
    colors = plt.cm.jet(np.linspace(0, 1, 10))

    # Plot raw data
    _ax00.text(1.05, 1., '(a)', horizontalalignment='center', verticalalignment='top', transform=_ax00.transAxes)
    _ax00 = plot_class(axis=_ax00, measurement_class=meas_data)
    _ax00.set_xlabel('Voltage [V]')

    # Plot peak locations
    _ax01.text(1.05, 1., '(b)', horizontalalignment='center', verticalalignment='top', transform=_ax01.transAxes)
    peak_collection = identify_peaks(meas_data=meas_data)
    _ax01 = plot_class(axis=_ax01, measurement_class=meas_data, alpha=0.2)
    for i, peak_data in enumerate(peak_collection):
        if peak_data.relevant:
            _ax01.plot(peak_data.get_x, peak_data.get_y, 'x', color='r', alpha=1)
    _ax01.set_xlabel('Voltage [V]')

    # Plot q mode separation and mode ordering
    _ax10.text(1.05, 1., '(c)', horizontalalignment='center', verticalalignment='top', transform=_ax10.transAxes)
    labeled_collection = LabeledPeakCollection(peak_collection)
    _ax10 = get_standard_axis(axis=_ax10)
    min_q = min(labeled_collection.q_dict.keys())
    mode_sequence_range = range(min_q, max(labeled_collection.q_dict.keys())+2)
    for i in mode_sequence_range:
        try:
            cluster_array, value_slice = labeled_collection.get_mode_sequence(long_mode=i)
            # Get normalized measurement
            x_sample, y_measure = labeled_collection.get_measurement_data_slice(union_slice=value_slice)
            _ax10.plot(x_sample, y_measure, alpha=1, color=colors[(cluster_array[0].get_longitudinal_mode_id - min_q) % len(colors)])
        except AttributeError:
            break
    for i, peak_data in enumerate(labeled_collection):
        if peak_data.relevant:
            _ax10.plot(peak_data.get_x, peak_data.get_y, 'x', color=colors[(peak_data.get_transverse_mode_id - min_q)  % len(colors)], alpha=1)
    _ax10.set_xlabel('Voltage [V]')

    # Plot finalized labeled peaks
    _ax11.text(1.05, 1., '(d)', horizontalalignment='center', verticalalignment='top', transform=_ax11.transAxes)
    meas_data = get_converted_measurement_data(meas_class=meas_data, q_offset=Q_OFFSET, verbose=False)
    labeled_collection = LabeledPeakCollection(identify_peaks(meas_data=meas_data), q_offset=Q_OFFSET)
    # _ax11 = plot_class(axis=_ax11, measurement_class=meas_data, alpha=0.2)
    # _ax11 = plot_cluster_collection(axis=_ax11, data=labeled_collection)
    min_q = min(labeled_collection.q_dict.keys())
    mode_sequence_range = range(min_q, max(labeled_collection.q_dict.keys())+2)
    for i in mode_sequence_range:
        try:
            cluster_array, value_slice = labeled_collection.get_mode_sequence(long_mode=i)
            # Get normalized measurement
            x_sample, y_measure = labeled_collection.get_measurement_data_slice(union_slice=value_slice)
            _ax11.plot(x_sample, y_measure, alpha=.2, color=colors[(cluster_array[0].get_longitudinal_mode_id - min_q) % len(colors)])
        except AttributeError:
            print(i, f'break out of mode sequence')
            break

    for cluster in labeled_collection.get_clusters:
        if cluster.get_transverse_mode_id == 0:
            plt.gca().set_prop_cycle(None)
        for peak_data in cluster:
            if peak_data.relevant:
                _ax11.plot(peak_data.get_x, peak_data.get_y, 'x', color=colors[(peak_data.get_transverse_mode_id - min_q) % len(colors)], alpha=1)
        _ax11.text(x=cluster.get_avg_x, y=cluster.get_max_y, s=f'({cluster.get_longitudinal_mode_id}, {cluster.get_transverse_mode_id})', fontsize=10, horizontalalignment='center', verticalalignment='bottom')
    _ax11 = get_standard_axis(axis=_ax11)
    _ax11.set_xlabel('Cavity Length [nm]')


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
