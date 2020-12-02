import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Union
from src.plot_npy import get_standard_axis, plot_class, plot_peak_collection, plot_cluster_collection
from src.import_data import SyncMeasData
from src.peak_relation import LabeledPeakCollection, get_value_to_data_slice, flatten_clusters


def plot_isolated_long_mode(axis: plt.axes, data_class: SyncMeasData, collection: LabeledPeakCollection, long_mode: int, trans_mode: Union[int, None]) -> plt.axis:
    # Plot array
    try:
        cluster_array, value_slice = collection.get_mode_sequence(long_mode=long_mode, trans_mode=trans_mode)
    except ValueError:
        logging.warning(f'Longitudinal mode {long_mode} not well defined')
        return axis

    data_class.slicer = get_value_to_data_slice(data_class=data_class, value_slice=value_slice)
    # Plot data (data_slice)
    axis = plot_class(axis=axis, measurement_class=data_class)
    # Plot cluster array
    axis = plot_cluster_collection(axis=axis, data=cluster_array)
    # # Plot peak array
    # axis = plot_peak_collection(axis=axis, data=flatten_clusters(data=cluster_array))
    return get_standard_axis(axis=axis)


def plot_3d_sequence(axis: plt.axis, data_classes: List[SyncMeasData], long_mode: int, trans_mode: Union[int, None]) -> plt.axis:
    # Set 3D plot
    axis = plt.subplot(projection='3d')
    # for i, data_class in enumerate(data_classes):
    #     axis.plot(measurement_class.x_data, measurement_class.y_data)
    return axis


if __name__ == '__main__':
    from src.peak_identifier import identify_peaks
    from src.plot_npy import prepare_measurement_plot

    # Reference files
    file_samp = 'samples_1s_10V_rate1300000.0'

    # Measurement files
    filename_base = 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration'
    filenames = [filename_base + str(i) for i in range(5)]
    meas_iterations = [SyncMeasData(meas_file=file_meas, samp_file=file_samp, scan_file=None) for file_meas in filenames]
    identified_peaks = [identify_peaks(meas_data=data) for data in meas_iterations]
    labeled_peaks = [LabeledPeakCollection(optical_mode_collection=collection) for collection in identified_peaks]

    # Test
    index = 0
    long_mode = 0

    ax_full, measurement_class = prepare_measurement_plot(filenames[index])
    ax_full = plot_class(axis=ax_full, measurement_class=measurement_class)

    fig, ax_array = plt.subplots(2, 2)
    ax_array = np.ndarray.flatten(ax_array)
    for long_mode in range(len(ax_array)):
        # Plot specific longitudinal mode
        ax_array[long_mode] = plot_isolated_long_mode(axis=ax_array[long_mode], data_class=meas_iterations[index], collection=labeled_peaks[index], long_mode=long_mode, trans_mode=2)
        if long_mode == 1:
            ax_array[long_mode].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot mode separation
        cluster_array, value_slice = labeled_peaks[index].get_mode_sequence(long_mode=long_mode)
        # Temporary coloring
        ax_full.axvline(x=value_slice[0], color=f'C{long_mode+1}')
        ax_full.axvline(x=value_slice[1], color=f'C{long_mode+1}', label=f'Long. Mode={long_mode}')

    ax_full.legend()
    plt.show()
