import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm  # For displaying for-loop process to console
from typing import Iterable, List, Tuple, Dict
from src.import_data import SyncMeasData
from src.plot_functions import get_standard_axis
from src.plot_iteration_mode import most_frequent
from src.peak_relation import LabeledPeakCollection, get_value_to_data_slice, find_nearest_index, LabeledPeakCluster, SAMPLE_WAVELENGTH, flatten_clusters


def pad_to_dense(nested_array: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Returns padded array and shape"""
    max_width = np.max([len(i) for i in nested_array])

    result = np.zeros((len(nested_array), max_width))  # Pre-load zero matrix
    for i, row in enumerate(nested_array):
        result[i, :len(row)] += row
    return result, (max_width, len(result))


def pad_to_pinned(nested_array: List[np.ndarray], pre_skip: List[int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Returns padded array and shape"""
    max_width = np.max([len(_array) + pre_skip[i] for i, _array in enumerate(nested_array)])

    result = np.zeros((len(nested_array), max_width))  # Pre-load zero matrix
    for i, row in enumerate(nested_array):
        skip_shift = pre_skip[i]
        result[i, skip_shift:(len(row)+skip_shift)] += row
    return result, (max_width, len(result))


def pad_slice(original: Tuple[int, int], additional: Tuple[int, int]) -> Tuple[int, int]:
    """Returns the Union slice of both original and additional slice"""
    return min(original[0], additional[0]), max(original[1], additional[1])


def store_peak_data(dictionary: Dict[Tuple[int, int, int], Tuple[List[float], List[int]]], cluster_array: List[LabeledPeakCluster], data_class: SyncMeasData) -> Dict[Tuple[int, int, int], Tuple[List[float], List[int]]]:
    """Returns dictionary in format: Dict[Tuple[long, trans, index], Tuple[List[pos_value], List[pos_index]]]"""
    for cluster in cluster_array:
        for k, peak in enumerate(cluster):
            key = (cluster.get_longitudinal_mode_id, cluster.get_transverse_mode_id, k)
            if key in dictionary:
                dictionary[key][0].append(peak.get_x)
                dictionary[key][1].append(find_nearest_index(array=data_class.x_boundless_data, value=peak.get_x))
            else:
                dictionary[key] = ([peak.get_x], [find_nearest_index(array=data_class.x_boundless_data, value=peak.get_x)])
    return dictionary


def plot_pinned_focus_side(collection_iterator: Iterable[LabeledPeakCollection], pin: Tuple[int, int]):
    """Create side-view plot which pins single modes"""

    # Filter consistent measurement peaks
    def filter_collection(iterator: Iterable[LabeledPeakCollection]) -> Iterable[LabeledPeakCollection]:
        """Includes filter for undetermined value slice and uncommon peak count"""
        result_tuple = []
        peak_count = []
        for i, _collection in tqdm(enumerate(iterator), desc=f'Pre-Processing'):
            try:
                # Pin value
                pin_cluster_array, _ = _collection.get_mode_sequence(long_mode=pin[0], trans_mode=pin[1])
            except IndexError:
                # specific mode could not be found
                continue
            count = len(flatten_clusters(data=pin_cluster_array))
            peak_count.append(count)
            result_tuple.append((count, _collection))
        # Filter count entries
        _most_freq_nr = most_frequent(peak_count)
        for (count, _collection) in result_tuple:
            if count == _most_freq_nr:
                yield _collection

    # Prepare parameters
    iter_count = 0
    trans_array = []
    total_pin_peak_index = []
    peak_dict = {}  # Dict[Tuple(long, trans, index), Tuple[List[pos_value], List[pos_index]]
    data_class_array = []
    total_value_slice_array = []
    total_data_slice_array = []
    for i, collection in tqdm(enumerate(filter_collection(collection_iterator)), desc=f'Process collections'):
        iter_count += 1  # Temp
        data_class = collection._get_data_class  # Access SyncMeasData class corresponding to peak data
        data_class_array.append(data_class)
        # Pin value
        pin_cluster_array, pin_value_slice = collection.get_mode_sequence(long_mode=pin[0], trans_mode=pin[1])
        # Store peak data
        peak_dict = store_peak_data(dictionary=peak_dict, cluster_array=pin_cluster_array, data_class=data_class)
        # Value to data slice
        total_value_slice_array.append(pin_value_slice)
        total_data_slice = get_value_to_data_slice(data_class=data_class, value_slice=pin_value_slice)
        total_data_slice_array.append(total_data_slice)  # Store data slice
        # Define focus peak index offset
        first_peak_offset = peak_dict[(pin[0], pin[1], 0)][1][-1] - total_data_slice[0]
        total_pin_peak_index.append(first_peak_offset)
        # Store transmission data
        x_array, y_array = np.asarray(collection.get_measurement_data_slice(union_slice=total_data_slice))
        trans_array.append(y_array)

    # Find data class corresponding to average length
    value_slice_length_array = [value_slice[1] - value_slice[0] for value_slice in total_value_slice_array]
    sorted_data_class_array = [data for _, data in sorted(zip(value_slice_length_array, data_class_array), key=lambda pair: pair[0])]  # rearange
    data_class_index = find_nearest_index(array=value_slice_length_array, value=np.max(value_slice_length_array))
    lead_data_class = sorted_data_class_array[data_class_index]

    # Clear pin shift
    total_pin_peak_index = np.asarray(total_pin_peak_index)
    pre_skip = total_pin_peak_index - np.min(total_pin_peak_index)
    # Align front of arrays
    trans_array = [_array[0:] for i, _array in enumerate(trans_array)]
    lead_index = trans_array.index(max(trans_array, key=len))
    index_offset = total_data_slice_array[lead_index][0]  # + pre_skip[lead_index]

    # Format data
    trans_array, array_shape = pad_to_dense(nested_array=trans_array)
    # Plot measurement data
    fig, ax = plt.subplots()
    for i, meas_array in enumerate(trans_array):
        color = 'b'
        if i == len(trans_array) - 1:
            color = 'darkorange'
        ax.plot(meas_array, color=color)
    # # Filter dict entries
    # Plot mean and std peak locations
    y_bot, y_top = ax.get_ylim()
    pos_text_height = y_bot + 0.95 * (y_top - y_bot)  # Data coordinates
    diff_text_upper_height = y_bot + 0.065 * (y_top - y_bot)
    diff_text_lower_height = y_bot + 0.062 * (y_top - y_bot)
    plot_peak_pos = {}
    ax.plot(np.NaN, np.NaN, '-', color='none', label=r'$\mu$, $\sigma$')
    for i in range(20):
        key = (pin[0], pin[1], i)
        if key in peak_dict:
            pos_array = peak_dict[key][0]
            # if len(pos_array) != most_freq_nr:
            #     continue
            # if len(pos_array) < 0.8 * iter_count:
            #     continue
            avg_pos_value = np.mean(pos_array)  # Sum / Count [nm]
            std_pos_value = np.std(pos_array)
            mean = round((avg_pos_value), 2)  # / (SAMPLE_WAVELENGTH / 2), 3)
            std = round((std_pos_value), 2)  # / (SAMPLE_WAVELENGTH / 2), 3)
            avg_pos_data = np.mean([peak_index - pre_skip[i] - total_data_slice_array[i][0] for i, peak_index in enumerate(peak_dict[key][1])])
            ax.axvline(x=avg_pos_data, color='r', ls='--', label=f'({i}): {mean}'+r'$\pm$'+f'{std} [nm]')
            # ax.text(x=avg_pos_data, y=pos_text_height, s=r'$\mu=$' + f' {mean},\n' + r'$\sigma=$ ' + f'{std} [nm]', horizontalalignment='center', fontsize=12)
            # ax.plot(np.NaN, np.NaN, '-', color='none', label=f'({i}): {mean}'+r'$\pm$'+f'{std} [nm]')
            ax.text(x=avg_pos_data, y=pos_text_height, s=f'({i})', horizontalalignment='center', fontsize=12)
            plot_peak_pos[i] = avg_pos_data
        else:
            break  # No higher index peaks available
    # Plot peak distances
    for i in range(20):
        key_first = (pin[0], pin[1], i)
        key_second = (pin[0], pin[1], i+1)
        if key_second in peak_dict and key_first in peak_dict:
            pos_array_first = peak_dict[key_first][0]
            pos_array_second = peak_dict[key_second][0]
            # if len(pos_array_first) != most_freq_nr or len(pos_array_second) != most_freq_nr:
            #     continue
            distances = np.asarray(pos_array_second) - np.asarray(pos_array_first)
            avg_dist_value = np.mean(distances)
            std_dist_value = np.std(distances)
            mean = round((avg_dist_value), 2)  # / (SAMPLE_WAVELENGTH / 2), 3)
            std = round((std_dist_value), 2)  # / (SAMPLE_WAVELENGTH / 2), 3)
            ax.plot(np.NaN, np.NaN, '-', color='none') #, label=f'({i}->{i+1}) : ' + r'$\mu=$' + f' {mean}' + r', $\sigma=$ ' + f'{std} [nm]')
            mid_peak_pos = int((plot_peak_pos[i+1] - plot_peak_pos[i])/2. + plot_peak_pos[i])
            text = ax.text(x=mid_peak_pos, y=diff_text_upper_height if i % 2 == 0 else diff_text_lower_height, s=f'({i}->{i+1})\n' + r'$\mu=$' + f' {mean} [nm]\n' + r'$\sigma=$ ' + f'{std} [nm]', horizontalalignment='center', fontsize=12)
            text.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='white'))
        else:
            break
    # Set plot layout
    locs, labels = plt.xticks()
    # xticks = [int(lead_data_class.x_boundless_data[int(index_offset + index)]) for index in locs[0:-1]]
    # print(xticks)

    # Successful averaging
    xticks = []
    std_ticks = []
    for index in locs[0:-1]:
        data_values = []
        for i, _data_class in enumerate(data_class_array):
            data_values.append(_data_class.x_boundless_data[int(total_data_slice_array[i][0] + index)])
        xticks.append(round(np.mean(data_values), 2))
        std_ticks.append(np.std(data_values))
    # print(xticks)

    plt.xticks(locs[0:-1], xticks)
    plt.xlim(locs[1], locs[-1])
    # pin = (7, 8)
    plt.title(f'Focus mode (q={pin[0]}, m+n={pin[1]}, {iter_count} iterations)')
    ax = get_standard_axis(axis=ax)
    ax.set_xlabel('Cavity Length ('+ r'$\pm$' + f'{round(np.mean(std_ticks), 1)}' + ') [nm]')  # Voltage [V]
    plt.grid(True)
    plt.legend(loc=1, prop={'size': 12})  # bbox_to_anchor=(1.05, 1), loc='upper left',
    # plt.tight_layout(pad=1.08, w_pad=0.1, h_pad=0.1, rect=(0, 0, 1, 1))


def plot_pinned_focus_top(collection_iterator: Iterable[LabeledPeakCollection], pin: Tuple[int, int], focus: List[Tuple[int, int]]):  #  -> plt.axes
    """Create color plot which pins on single mode and displays one or more reference modes"""

    # Filter consistent measurement peaks
    def filter_collection(iterator: Iterable[LabeledPeakCollection]) -> Iterable[LabeledPeakCollection]:
        """Includes filter for undetermined value slice and uncommon peak count"""
        result_tuple = []
        peak_count = []
        for i, _collection in tqdm(enumerate(iterator), desc=f'Pre-Processing'):
            try:
                # Pin value
                _collection.get_mode_sequence(long_mode=pin[0], trans_mode=pin[1])
                # for j, sub_focus in enumerate(focus):
                #     _collection.get_mode_sequence(long_mode=sub_focus[0], trans_mode=sub_focus[1])
            except IndexError:
                # specific mode could not be found
                continue
            yield _collection
            # count = len(flatten_clusters(data=pin_cluster_array))
            # peak_count.append(count)
            # result_tuple.append((count, _collection))

        # # Filter count entries
        # _most_freq_nr = most_frequent(peak_count)
        # for (count, _collection) in result_tuple:
        #     if count == _most_freq_nr:
        #         yield _collection

    # Prepare parameters
    iter_count = 0
    peak_dict = {}  # Dict[Tuple(long, trans, index), Tuple[List[pos_value], List[pos_index]]
    total_value_slice_array = []
    total_data_slice_array = []
    trans_array = []
    value_focus_array = []
    data_focus_array = []
    data_class_array = []
    total_pin_peak_index = []
    for i, collection in tqdm(enumerate(filter_collection(collection_iterator)), desc=f'Process collections'):
        iter_count += 1  # Temp
        data_class = collection._get_data_class  # Access SyncMeasData class corresponding to peak data
        # Pin value
        pin_cluster_array, pin_value_slice = collection.get_mode_sequence(long_mode=pin[0], trans_mode=pin[1])
        # Store peak data
        peak_dict = store_peak_data(dictionary=peak_dict, cluster_array=pin_cluster_array, data_class=data_class)

        # Get corresponding data_slice (index slice) and pin index
        # Focus value
        for j, sub_focus in enumerate(focus):
            try:
                foc_cluster_array, foc_value_slice = collection.get_mode_sequence(long_mode=sub_focus[0], trans_mode=sub_focus[1])
            except IndexError:
                # specific mode could not be found
                continue
            # Store peak data
            peak_dict = store_peak_data(dictionary=peak_dict, cluster_array=foc_cluster_array, data_class=data_class)
            # Get data slice
            foc_data_slice = get_value_to_data_slice(data_class=data_class, value_slice=foc_value_slice)

            # Store value slice
            # print(sub_focus, foc_value_slice)
            if len(value_focus_array) <= j:
                value_focus_array.insert(j, foc_value_slice)
                data_focus_array.insert(j, foc_data_slice)
            else:
                value_focus_array[j] = pad_slice(original=value_focus_array[j], additional=foc_value_slice)
                data_focus_array[j] = pad_slice(original=data_focus_array[j], additional=foc_data_slice)

        # total min-max value slice based on pin and focus slices
        total_value_slice = pin_value_slice
        for value_bound in value_focus_array:
            total_value_slice = pad_slice(original=total_value_slice, additional=value_bound)
        # Value to data slice
        total_data_slice = get_value_to_data_slice(data_class=data_class, value_slice=total_value_slice)
        # Define focus peak index offset
        # TODO: Get largest peak position
        key_height_tuple = [((pin[0], pin[1], k), peak.get_y) for k, peak in enumerate(flatten_clusters(pin_cluster_array))]
        sorted_key_height_tuple = list(sorted(key_height_tuple, key=lambda pair: pair[1]))
        first_peak_offset = peak_dict[sorted_key_height_tuple[-1][0]][1][-1] - total_data_slice[0]
        total_pin_peak_index.append(first_peak_offset)

        total_value_slice_array.append(total_value_slice)  # Store value slice
        total_data_slice_array.append(total_data_slice)  # Store data slice
        trans_array.append(np.asarray(collection.get_measurement_data_slice(union_slice=total_data_slice)[1]))  # Store y-data
        # Store data class of largest array len
        data_class_array.append(data_class)

    # Find data class corresponding to average length
    value_slice_length_array = [value_slice[1] - value_slice[0] for value_slice in total_value_slice_array]
    sorted_data_class_array = [data for _, data in sorted(zip(value_slice_length_array, data_class_array), key=lambda pair: pair[0])]  # rearange
    data_class_index = find_nearest_index(array=value_slice_length_array, value=np.max(value_slice_length_array))
    lead_data_class = sorted_data_class_array[data_class_index]

    # Clear pin shift
    total_pin_peak_index = np.asarray(total_pin_peak_index)
    min_data_offset = np.min(total_pin_peak_index)
    pre_skip = np.max(total_pin_peak_index) - total_pin_peak_index  # - min_data_offset)
    # Align front of arrays
    # trans_array = [_array[pre_skip[i]:] for i, _array in enumerate(trans_array)]

    # Format data
    lead_index = trans_array.index(max(trans_array, key=len))
    index_offset = total_data_slice_array[lead_index][0]  # + pre_skip[lead_index]
    trans_array, array_shape = pad_to_pinned(nested_array=trans_array, pre_skip=pre_skip)  # , dictionary=peak_dict, key=(pin[0], pin[1], 0))
    trans_array = np.transpose(trans_array)  # Transpose

    # Define font
    font_size = 22
    plt.rcParams.update({'font.size': font_size})

    # Total slice
    x, y = np.mgrid[0:array_shape[0], 0:array_shape[1]]
    plt.pcolormesh(x, y, trans_array, norm=colors.LogNorm())

    locs, labels = plt.xticks()
    xticks = [int(lead_data_class.x_boundless_data[int(index_offset + index)]) for index in locs[0:-1]]
    plt.xticks(locs[0:-1], xticks)
    plt.title(f'Pinned plot over {iter_count} sample iterations')
    plt.ylabel(f'Different Iterations [a.u.]', fontsize=font_size)
    plt.xlabel(f'Cavity length (based on average) [nm]', fontsize=font_size)
    plt.grid(True)

    # for i, value_bound in enumerate(value_focus_array):
    #     data_bound = get_value_to_data_slice(data_class=lead_data_class, value_slice=value_bound)  # Sample to index
    #     # data_bound = data_focus_array[i]
    #     fig, ax = plt.subplots()
    #     array_bound = (max(data_bound[0] - index_offset, 0), (data_bound[1] - index_offset))
    #
    #     focus_trans_array = trans_array[array_bound[0]:array_bound[1]]  # After transpose
    #     focus_array_shape = focus_trans_array.shape
    #     # Focus slice
    #     x, y = np.mgrid[0:focus_array_shape[0], 0:focus_array_shape[1]]
    #     ax.pcolormesh(x, y, focus_trans_array, norm=colors.LogNorm())
    #
    #     # Plot peak average position
    #     rev_key = (pin[0], pin[1], 0)
    #     rev_pos_value = np.mean(peak_dict[rev_key][0])  # Sum / Count [nm]
    #     long_mode, trans_mode = focus[i]
    #     for j in range(10):
    #         key = (long_mode, trans_mode, j)
    #         if key in peak_dict:
    #             if len(peak_dict[key][0]) < 0.5 * iter_count:
    #                 continue
    #             avg_pos_value = np.mean(peak_dict[key][0])  # Sum / Count [nm]
    #             std_pos_value = np.std(peak_dict[key][0])
    #             mean = round((avg_pos_value - rev_pos_value)/(SAMPLE_WAVELENGTH/2), 3)
    #             std = round((std_pos_value)/(SAMPLE_WAVELENGTH/2), 3)
    #             avg_pos_data = np.mean([peak_index - pre_skip[k] - data_bound[0] for k, peak_index in enumerate(peak_dict[key][1])])
    #             ax.axvline(x=avg_pos_data, color='r', ls='--', label=f'{mean}' + r' $\pm$ ' + f'{std}' + r' [$\lambda / 2$]')
    #         else:
    #             break  # No higher index peaks available
    #
    #     locs, labels = plt.xticks()
    #     xticks = [int(lead_data_class.x_boundless_data[int(index_offset + index)]) for index in locs[0:-1]]
    #     plt.xticks(locs[0:-1], xticks)
    #     plt.title(f'Focus mode (q={focus[i][0]}, m+n={focus[i][1]}). Transmission relative to fundamental mode')
    #     ax = get_standard_axis(axis=ax)
    #     # plt.ylabel(f'Different Iterations [a.u.]', fontsize=font_size)
    #     # plt.xlabel(f'Cavity length [nm]', fontsize=font_size)
    #     plt.grid(True)
    #     plt.legend()


def plot_pinned_track_top(collection_iterator: Iterable[LabeledPeakCollection], trans_mode_track: int):
    """Create color plot that tracks a single transverse mode through spectrum"""

    # Filter consistent measurement peaks
    def filter_collection(iterator: Iterable[LabeledPeakCollection]) -> Iterable[LabeledPeakCollection]:
        """Includes filter for undetermined value slice and uncommon peak count"""
        for i, _collection in tqdm(enumerate(iterator), desc=f'Pre-Processing'):
            yield _collection

    # Prepare parameters
    iter_count = -1
    peak_dict = {}  # Dict[Tuple(long, trans, index), Tuple[List[pos_value], List[pos_index]]
    total_value_slice_array = []
    total_data_slice_array = []
    trans_array = []
    data_class_array = []
    total_pin_peak_index = []
    for i, collection in tqdm(enumerate(filter_collection(collection_iterator)), desc=f'Process collections'):
        data_class = collection._get_data_class  # Access SyncMeasData class corresponding to peak data

        # Get corresponding data_slice (index slice) and pin index
        # 7th: for long_mode, trans_mode_track in [(8, 7), (9, 7), (10, 7), (11, 7), (13, 0), (13, 7)]:
        # 8th: for long_mode, trans_mode_track in [(8, 8), (9, 8), (10, 8), (12, 1), (12, 7), (13, 8)]:
        peak_tracker = [6, 2, 2, 3, 4, 0, 0, 0, 0, 0, 0]
        if i == 0:
            track_coords = [(5, 5), (6, 9), (8, 2), (9, 8), (10, 8)]
        elif i == 1:
            track_coords = [(8, 8), (9, 8), (10, 8), (12, 1), (12, 7), (13, 8)]
        else:
            track_coords = [(8, 8), (9, 8), (10, 8), (12, 1), (12, 7), (13, 8)]

        for long_mode, trans_mode_track in track_coords:  #long_mode in range(collection.get_min_q_id, collection.get_max_q_id):
            iter_count += 1  # Temp
            try:
                pin_cluster_array, total_value_slice = collection.get_mode_sequence(long_mode=long_mode, trans_mode=trans_mode_track)
            except IndexError:
                print(f'Something happend')
                # specific mode could not be found
                continue
            # Store peak data
            peak_dict = store_peak_data(dictionary=peak_dict, cluster_array=pin_cluster_array, data_class=data_class)
            # Get data slice
            total_data_slice = get_value_to_data_slice(data_class=data_class, value_slice=total_value_slice)
            print(long_mode, trans_mode_track)

            # # TODO: Get largest peak position
            # key_height_tuple = [((long_mode, trans_mode_track, k), peak.get_y) for k, peak in enumerate(flatten_clusters(pin_cluster_array))]
            # sorted_key_height_tuple = list(sorted(key_height_tuple, key=lambda pair: pair[1]))
            # if trans_mode_track == 0:
            #     print('true')
            #     first_peak_offset = peak_dict[sorted_key_height_tuple[-6][0]][1][-1] - total_data_slice[0]
            # elif long_mode in [8, 9, 10]:
            #     first_peak_offset = peak_dict[(long_mode, trans_mode_track, 0)][1][-1] - total_data_slice[0]
            # else:
            # first_peak_offset = peak_dict[sorted_key_height_tuple[0][0]][1][-1] - total_data_slice[0]
            # last_peak_index = len(pin_cluster_array) - 1
            # first_peak_offset = peak_dict[(long_mode, trans_mode_track, last_peak_index)][1][-1] - total_data_slice[0]
            # first_peak_offset = find_nearest_index(array=data_class.x_boundless_data, value=pin_cluster_array[0].get_avg_x) - total_data_slice[0]
            peak_index = peak_tracker[iter_count]
            first_peak_offset = peak_dict[(long_mode, trans_mode_track, peak_index)][1][-1] - total_data_slice[0]
            total_pin_peak_index.append(first_peak_offset)

            total_value_slice_array.append(total_value_slice)  # Store value slice
            total_data_slice_array.append(total_data_slice)  # Store data slice
            trans_array.append(np.asarray(collection.get_measurement_data_slice(union_slice=total_data_slice)[1]))  # Store y-data
            # Store data class of largest array len
            data_class_array.append(data_class)

    # Find data class corresponding to average length
    value_slice_length_array = [value_slice[1] - value_slice[0] for value_slice in total_value_slice_array]
    sorted_data_class_array = [data for _, data in sorted(zip(value_slice_length_array, data_class_array), key=lambda pair: pair[0])]  # rearange
    data_class_index = find_nearest_index(array=value_slice_length_array, value=np.max(value_slice_length_array))
    lead_data_class = sorted_data_class_array[data_class_index]

    # Clear pin shift
    total_pin_peak_index = np.asarray(total_pin_peak_index)
    pre_skip = np.max(total_pin_peak_index) - total_pin_peak_index

    # Format data
    lead_index = trans_array.index(max(trans_array, key=len))
    index_offset = total_data_slice_array[lead_index][0]
    trans_array, array_shape = pad_to_pinned(nested_array=trans_array, pre_skip=pre_skip)
    trans_array = np.transpose(trans_array)  # Transpose

    # Define font
    font_size = 22
    plt.rcParams.update({'font.size': font_size})
    _, _ax = plt.subplots()

    # Total slice
    x, y = np.mgrid[0:array_shape[0]+1, 0:array_shape[1]+1]
    _ax.pcolormesh(x, y, trans_array, norm=colors.LogNorm())

    ylocs = _ax.get_yticks()
    ylocs = np.linspace(start=ylocs[0], stop=ylocs[-1], num=13)
    yticks = [f'{i+3}' for i, label in enumerate(ylocs[0:-1])]
    plt.yticks(ylocs[0:-1], yticks)

    xlocs = _ax.get_xticks()
    xlocs = np.linspace(start=xlocs[0], stop=xlocs[-1], num=9)
    xticks = [int(lead_data_class.x_boundless_data[int(index_offset + index)]) for index in xlocs[0:-1]]
    mid_tick = xticks[0]  # (xticks[-1] - xticks[0]) / 2 + xticks[0]
    xticks = [str(tick-mid_tick) for tick in xticks]
    plt.xticks(xlocs[0:-1], xticks)
    # ylocs, _ = plt.yticks()
    # _ax.set_yticks(ylocs[0:-1], [i+8 for i, index in enumerate(ylocs[0:-1])])
    _ax.set_title(f'{trans_mode_track}th transverse mode over multiple FSR')
    _ax.set_ylabel(f'# Fundamental mode', fontsize=font_size)
    _ax.set_xlabel(f'Mode width [nm]', fontsize=font_size)
    # _ax.get_xaxis().set_visible(False)
    # _ax.grid(True)
