import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Union
from src.plot_functions import get_standard_axis, plot_class, plot_peak_collection, plot_cluster_collection
from src.import_data import SyncMeasData, FileToMeasData
from src.peak_identifier import identify_peaks
from src.peak_relation import LabeledPeakCollection, get_value_to_data_slice, flatten_clusters, get_slice_range, get_converted_measurement_data


def plot_isolated_long_mode(axis: plt.axes, data_class: SyncMeasData, collection: LabeledPeakCollection, long_mode: int, trans_mode: Union[int, None], **kwargs) -> plt.axis:
    # Plot array
    try:
        cluster_array, value_slice = collection.get_mode_sequence(long_mode=long_mode, trans_mode=trans_mode)
    except ValueError:
        logging.warning(f'Longitudinal mode {long_mode} not well defined')
        return axis

    data_class.slicer = get_value_to_data_slice(data_class=data_class, value_slice=value_slice)
    # Plot data (data_slice)
    axis = plot_class(axis=axis, measurement_class=data_class, **kwargs)
    # Plot cluster array
    axis = plot_cluster_collection(axis=axis, data=cluster_array)
    # # Plot peak array
    # axis = plot_peak_collection(axis=axis, data=flatten_clusters(data=cluster_array))
    return get_standard_axis(axis=axis)


def whole_integer_divider(num: int, div: int) -> List[int]:
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


def plot_3d_sequence(data_classes: List[SyncMeasData], long_mode: int, trans_mode: Union[int, None]) -> plt.axis:
    # Set 3D plot
    fig = plt.figure()
    axis = fig.gca(projection='3d')

    # Store slices to ensure equally size arrays
    cluster_arrays = []
    q_mode_peaks = []
    data_slices = []
    data_slices_range = []
    for data_class in data_classes:
        collection = LabeledPeakCollection(identify_peaks(data_class))
        try:
            cluster_array, value_slice = collection.get_mode_sequence(long_mode=long_mode, trans_mode=trans_mode)
            cluster_arrays.append(cluster_array)
            q_mode_peaks.append(collection.q_dict[long_mode])
        except ValueError:
            logging.warning(f'Longitudinal mode {long_mode} not well defined')
            return axis
        data_slice = get_value_to_data_slice(data_class=data_class, value_slice=value_slice)
        data_slices.append(data_slice)
        data_slices_range.append(get_slice_range(data_slice))  # Store range

    # Prepare plot data
    leading_index = data_slices_range.index(max(data_slices_range))
    leading_slice = data_slices[leading_index]
    for i, slice in enumerate(data_slices):
        range_diff = get_slice_range(leading_slice) - get_slice_range(slice)
        padding = whole_integer_divider(num=range_diff, div=2)
        data_slices[i] = (slice[0] - padding[0], slice[1] + padding[1])

    sliced_xs = data_classes[leading_index].x_boundless_data[leading_slice[0]: leading_slice[1]]
    xs = np.arange(get_slice_range(leading_slice))  # sliced_xs  #
    zs = np.arange(len(data_slices))
    verts = []
    peaks = []
    for i, slice in enumerate(data_slices):
        data_class = data_classes[i]
        data_class.slicer = slice
        ys = data_class.y_data
        verts.append(list(zip(xs, ys)))
        # Peak scatter plot
        peak_list = flatten_clusters(data=cluster_arrays[i])
        peaks.append(peak_list)  # Collect peaks for polarisation cross section
        yp = [peak.get_y for peak in peak_list]
        xp = [peak.get_relative_index for peak in peak_list]
        zp = [zs[i] for j in range(len(peak_list))]
        axis.scatter(xp, zp, yp, marker='o')

    # Draw individual measurement polygons
    poly = PolyCollection(verts)
    poly.set_alpha(.7)
    axis.add_collection3d(poly, zs=zs, zdir='y')

    # Draw polarisation cross section
    cross_section_count = len(peaks[0])
    if all(len(peak_array) == cross_section_count for peak_array in peaks):  # Able to build consistent cross sections
        cross_peaks = list(map(list, zip(*peaks)))  # Transposes peaks-list to allow for cross section ordering
        xc = []
        # Insert 0 bound values
        zc = list(zs)
        zc.insert(0, zc[0])
        zc.append(zc[-1])
        peak_verts = []
        face_colors = [[v, .3, .3] for v in np.linspace(.5, 1., len(cross_peaks))]
        for i, cross_section in enumerate(cross_peaks):
            yc = [peak.get_y for peak in cross_section]
            # Insert 0 bound values
            yc.insert(0, 0)
            yc.append(0)
            xc.append(int(np.mean([peak.get_relative_index for peak in cross_section])))  # np.mean([peak.get_x for peak in cross_section]))  #
            peak_verts.append(list(zip(zc, yc)))

            poly = PolyCollection([list(zip(zc, yc))])  # peak_verts
            poly.set_alpha(1)
            poly.set_facecolor(face_colors[i])
            axis.add_collection3d(poly, zs=xc[-1], zdir='x')

        # poly = PolyCollection(peak_verts)
        # poly.set_alpha(1)
        # axis.add_collection3d(poly, zs=xc, zdir='x')
        print('plotting')
    else:
        logging.warning(f'Cross section (peak) count is not consistent')

    axis.set_xlabel('Relative cavity length [nm]')
    axis.set_xlim3d(0, len(xs))
    # axis.set_xticks(xs)
    axis.set_ylabel('Polarisation [10 Degree]')  # 'Measurement iterations')
    axis.set_ylim3d(-1, len(zs) + 1)
    # axis.set_yticks([str(10 * angle) for angle in zs])
    axis.set_zlabel('Transmission [a.u.]')
    axis.set_zlim3d(0, 1)
    # Set viewport
    axis.view_init(elev=22, azim=-15)
    return axis


if __name__ == '__main__':
    from src.plot_functions import prepare_measurement_plot

    # Reference files
    file_samp = 'samples_0_3s_10V_rate1300000.0'  # 'samples_1s_10V_rate1300000.0'

    # Measurement files
    # filenames = ['transrefl_hene_1s_10V_PMT5_rate1300000.0itteration{}'.format(i) for i in range(10)]
    filenames = ['transrefl_hene_0_3s_10V_PMT4_rate1300000.0itteration1_pol{:0=2d}0'.format(i) for i in range(19)]

    meas_iterations = [get_converted_measurement_data(FileToMeasData(meas_file=file_meas, samp_file=file_samp)) for file_meas in filenames]
    identified_peaks = [identify_peaks(meas_data=data) for data in meas_iterations]
    labeled_peaks = [LabeledPeakCollection(transmission_peak_collection=collection) for collection in identified_peaks]

    trans_mode = 2
    long_mode = 0
    plot_3d_sequence(data_classes=meas_iterations, long_mode=long_mode, trans_mode=trans_mode)

    def plot_cross_sections():
        # Test
        index = 0

        ax_full, measurement_class = prepare_measurement_plot(filenames[index])
        measurement_class = get_converted_measurement_data(measurement_class)
        ax_full = plot_class(axis=ax_full, measurement_class=measurement_class)

        fig, ax_array = plt.subplots(3, 3)
        ax_array = np.ndarray.flatten(ax_array)
        for index in range(len(ax_array)):
            # Plot specific longitudinal mode
            ax_array[index] = plot_isolated_long_mode(axis=ax_array[index], data_class=meas_iterations[index], collection=labeled_peaks[index], long_mode=long_mode, trans_mode=trans_mode)
            if index == 2:
                ax_array[index].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Plot mode separation
            cluster_array, value_slice = labeled_peaks[index].get_mode_sequence(long_mode=long_mode)
            # Temporary coloring
            ax_full.axvline(x=value_slice[0], color=f'C{long_mode+1}')
            ax_full.axvline(x=value_slice[1], color=f'C{long_mode+1}', label=f'Long. Mode={long_mode}')

        ax_full.legend()

    plot_cross_sections()

    plt.show()
