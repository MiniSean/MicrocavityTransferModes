import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Union
from src.plot_npy import get_standard_axis, plot_class, plot_peak_collection, plot_cluster_collection
from src.import_data import SyncMeasData
from src.peak_identifier import identify_peaks
from src.peak_relation import LabeledPeakCollection, get_value_to_data_slice, flatten_clusters, get_slice_range


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


def plot_3d_sequence(data_classes: List[SyncMeasData], long_mode: int, trans_mode: Union[int, None]) -> plt.axis:
    # Set 3D plot
    fig = plt.figure()
    axis = fig.gca(projection='3d')

    # Store slices to ensure equally size arrays
    cluster_arrays = []
    data_slices = []
    data_slices_range = []
    for data_class in data_classes:
        collection = LabeledPeakCollection(identify_peaks(data_class))
        try:
            cluster_array, value_slice = collection.get_mode_sequence(long_mode=long_mode, trans_mode=trans_mode)
            cluster_arrays.append(cluster_array)
        except ValueError:
            logging.warning(f'Longitudinal mode {long_mode} not well defined')
            return axis
        data_slice = get_value_to_data_slice(data_class=data_class, value_slice=value_slice)
        data_slices.append(data_slice)
        data_slices_range.append(get_slice_range(data_slice))  # Store range

    def whole_integer_divider(num: int, div: int) -> List[int]:
        return [num // div + (1 if x < num % div else 0) for x in range (div)]

    # Prepare plot data
    leading_slice = data_slices[data_slices_range.index(max(data_slices_range))]
    for i, slice in enumerate(data_slices):
        range_diff = get_slice_range(leading_slice) - get_slice_range(slice)
        padding = whole_integer_divider(num=range_diff, div=2)
        data_slices[i] = (slice[0] - padding[0], slice[1] + padding[1])

    xs = np.arange(get_slice_range(leading_slice))
    zs = np.arange(len(data_slices))
    verts = []
    for i, slice in enumerate(data_slices):
        data_class = data_classes[i]
        data_class.slicer = slice
        ys = data_class.y_data
        verts.append(list(zip(xs, ys)))
        # Peak scatter plot
        peak_list = flatten_clusters(data=cluster_arrays[i])
        yp = [peak.get_y for peak in peak_list]
        xp = [peak.get_relative_index for peak in peak_list]
        zp = [zs[i] for j in range(len(peak_list))]
        axis.scatter(xp, zp, yp, marker='o')

    poly = PolyCollection(verts)
    poly.set_alpha(1)
    axis.add_collection3d(poly, zs=zs, zdir='y')

    axis.set_xlabel('Sliced Measurement index [a.u.]')
    axis.set_xlim3d(0, len(xs))
    axis.set_ylabel('Measurement iterations')
    axis.set_ylim3d(-1, len(zs) + 1)
    axis.set_zlabel('Transmission [a.u.]')
    axis.set_zlim3d(0, 1)
    # Set viewport
    axis.view_init(elev=22, azim=-15)
    return axis


if __name__ == '__main__':
    from src.plot_npy import prepare_measurement_plot

    # Reference files
    file_samp = 'samples_1s_10V_rate1300000.0'

    # Measurement files
    # filenames = ['transrefl_hene_1s_10V_PMT5_rate1300000.0itteration{}'.format(i) for i in range(10)]
    filenames = ['transrefl_hene_1s_10V_PMT5_rate1300000.0_pol{:0=2d}0'.format(i) for i in range(19)]

    meas_iterations = [SyncMeasData(meas_file=file_meas, samp_file=file_samp, scan_file=None) for file_meas in filenames]
    identified_peaks = [identify_peaks(meas_data=data) for data in meas_iterations]
    labeled_peaks = [LabeledPeakCollection(optical_mode_collection=collection) for collection in identified_peaks]

    plot_3d_sequence(data_classes=meas_iterations, long_mode=0, trans_mode=1)

    def plot_cross_sections():
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

    # plot_cross_sections()

    plt.show()
