import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm  # For displaying for-loop process to console
from typing import Iterable, List, Tuple, Dict
from src.import_data import SyncMeasData
from src.allan_variance_analysis import file_fetch_function
from src.peak_relation import LabeledPeakCollection, get_value_to_data_slice, find_nearest_index, LabeledPeakCluster, SAMPLE_WAVELENGTH


def pad_to_dense(nested_array: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Returns padded array and shape"""
    max_width = np.max([len(i) for i in nested_array])

    result = np.zeros((len(nested_array), max_width))  # Pre-load zero matrix
    for i, row in enumerate(nested_array):
        result[i, :len(row)] += row
    return result, (max_width, len(result))


def pad_to_pinned(nested_array: List[np.ndarray], dictionary: Dict[Tuple[int, int, int], Tuple[List[float], List[int]]], key: Tuple[int, int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Returns padded array and shape"""
    value_array = np.asarray(dictionary[key][0])  # List[float]
    data_array = np.asarray(dictionary[key][1])  # List[int]
    min_data_offset = np.min(data_array)
    pre_skip = min_data_offset - data_array
    # Align front of arrays
    # nested_array = [_array[pre_skip[i]:] for i, _array in enumerate(nested_array)]

    return pad_to_dense(nested_array=nested_array)


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

    # Prepare parameters
    iter_count = 0
    trans_array = []
    total_pin_peak_index = []
    peak_dict = {}  # Dict[Tuple(long, trans, index), Tuple[List[pos_value], List[pos_index]]
    data_class_array = []
    total_value_slice_array = []
    total_data_slice_array = []
    for i, collection in tqdm(enumerate(collection_iterator), desc=f'Process collections'):
        iter_count = i + 1  # Temp
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
    trans_array = [_array[pre_skip[i]:] for i, _array in enumerate(trans_array)]
    min_index = trans_array.index(min(trans_array, key=len))
    lead_index = trans_array.index(max(trans_array, key=len))
    mid_len_diff = int((len(trans_array[lead_index]) - len(trans_array[min_index])) / 2)
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
    # Plot mean and std peak locations
    for i in range(10):
        key = (pin[0], pin[1], i)
        if key in peak_dict:
            if len(peak_dict[key][0]) < 0.8 * iter_count:
                continue
            avg_pos_value = np.mean(peak_dict[key][0])  # Sum / Count [nm]
            std_pos_value = np.std(peak_dict[key][0])
            mean = round((avg_pos_value), 2)  # / (SAMPLE_WAVELENGTH / 2), 3)
            std = round((std_pos_value), 2)  # / (SAMPLE_WAVELENGTH / 2), 3)

            avg_pos_data = np.mean([peak_index - pre_skip[i] - total_data_slice_array[i][0] for i, peak_index in enumerate(peak_dict[key][1])])
            print(key, avg_pos_value)

            ax.axvline(x=avg_pos_data, color='r', ls='--', label=r'$\mu=$' + f' {mean}' + r', $\sigma=$ ' + f'{std} [nm]')
        else:
            break  # No higher index peaks available

    # Set plot layout
    locs, labels = plt.xticks()
    xticks = [int(lead_data_class.x_boundless_data[int(index_offset + index)]) for index in locs[0:-1]]
    plt.xticks(locs[0:-1], xticks)
    font_size = 22
    plt.title(f'Focus mode (q={pin[0]}, m+n={pin[1]}, {iter_count} iterations)')
    plt.ylabel(f'Transmission [a.u.]', fontsize=font_size)
    #  with {round(std_value_slice[0], 2)}, {round(std_value_slice[1], 2)} ' + r'$\sigma$' + '
    plt.xlabel(f'Cavity length (based on average) [nm]', fontsize=font_size)
    plt.yscale('log')
    plt.grid(True)
    plt.legend()


def plot_pinned_focus_top(collection_iterator: Iterable[LabeledPeakCollection], pin: Tuple[int, int], focus: List[Tuple[int, int]]):  #  -> plt.axes
    """Create color plot which pins on single mode and displays one or more reference modes"""

    # Prepare parameters
    iter_count = 0
    largest_array_len = 0
    lead_data_class = None
    peak_dict = {}  # Dict[Tuple(long, trans, index), Tuple[List[pos_value], List[pos_index]]
    total_value_slice_array = []
    total_data_slice_array = []
    trans_array = []
    value_focus_array = []
    data_focus_array = []
    data_class_array = []
    total_pin_peak_index = []
    for i, collection in tqdm(enumerate(collection_iterator), desc=f'Process collections'):
        iter_count = i + 1  # Temp
        data_class = collection._get_data_class  # Access SyncMeasData class corresponding to peak data
        # Pin value
        pin_cluster_array, pin_value_slice = collection.get_mode_sequence(long_mode=pin[0], trans_mode=pin[1])
        # Store peak data
        peak_dict = store_peak_data(dictionary=peak_dict, cluster_array=pin_cluster_array, data_class=data_class)

        # Get corresponding data_slice (index slice) and pin index
        # Focus value
        for j, sub_focus in enumerate(focus):
            foc_cluster_array, foc_value_slice = collection.get_mode_sequence(long_mode=sub_focus[0], trans_mode=sub_focus[1])
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
        first_peak_offset = peak_dict[(pin[0], pin[1], 0)][1][-1] - total_data_slice[0]
        total_pin_peak_index.append(first_peak_offset)

        total_value_slice_array.append(total_value_slice)  # Store value slice
        total_data_slice_array.append(total_data_slice)  # Store data slice
        # data_pin_array.append(data_pin)  # Store data pin
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
    pre_skip = total_pin_peak_index - min_data_offset
    # Align front of arrays
    trans_array = [_array[pre_skip[i]:] for i, _array in enumerate(trans_array)]

    # Format data
    min_index = trans_array.index(min(trans_array, key=len))
    lead_index = trans_array.index(max(trans_array, key=len))
    mid_len_diff = int((len(trans_array[lead_index]) - len(trans_array[min_index]))/2)
    index_offset = total_data_slice_array[lead_index][0]  # + pre_skip[lead_index]
    trans_array, array_shape = pad_to_dense(nested_array=trans_array)  # , dictionary=peak_dict, key=(pin[0], pin[1], 0))
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
    plt.title(f'Pinned plot over {iter_count} sample iterations (Second FSR)')
    plt.ylabel(f'Different Iterations [a.u.]', fontsize=font_size)
    #  with {round(std_value_slice[0], 2)}, {round(std_value_slice[1], 2)} ' + r'$\sigma$' + '
    plt.xlabel(f'Cavity length (based on average) [nm]', fontsize=font_size)
    plt.grid(True)

    for i, value_bound in enumerate(value_focus_array):
        data_bound = get_value_to_data_slice(data_class=lead_data_class, value_slice=value_bound)  # Sample to index
        # data_bound = data_focus_array[i]
        fig, ax = plt.subplots()
        array_bound = (max(data_bound[0] - index_offset, 0), (data_bound[1] - index_offset))

        focus_trans_array = trans_array[array_bound[0]:array_bound[1]]  # After transpose
        focus_array_shape = focus_trans_array.shape
        # Focus slice
        x, y = np.mgrid[0:focus_array_shape[0], 0:focus_array_shape[1]]
        ax.pcolormesh(x, y, focus_trans_array, norm=colors.LogNorm())

        # Plot peak average position
        rev_key = (pin[0], pin[1], 0)
        rev_pos_value = np.mean(peak_dict[rev_key][0])  # Sum / Count [nm]
        long_mode, trans_mode = focus[i]
        for j in range(10):
            key = (long_mode, trans_mode, j)
            if key in peak_dict:
                if len(peak_dict[key][0]) < 0.5 * iter_count:
                    continue
                avg_pos_value = np.mean(peak_dict[key][0])  # Sum / Count [nm]
                std_pos_value = np.std(peak_dict[key][0])
                mean = round((avg_pos_value - rev_pos_value)/(SAMPLE_WAVELENGTH/2), 3)
                std = round((std_pos_value)/(SAMPLE_WAVELENGTH/2), 3)

                # avg_pos_data = int(np.mean([find_nearest_index(array=data_class.x_boundless_data, value=peak_dict[key][0][i]) for i, data_class in enumerate(data_class_array)])) - data_bound[0]
                # avg_pos_data = int(np.mean([find_nearest_index(array=lead_data_class.x_boundless_data, value=peak) for peak in peak_dict[key][0]])) - data_bound[0]
                # avg_pos_data = find_nearest_index(array=lead_data_class.x_boundless_data, value=avg_pos_value) - data_bound[0]
                # avg_pos_data = int(np.mean(peak_dict[key][1])) - data_bound[0]
                avg_pos_data = np.mean([peak_index - pre_skip[k] - data_bound[0] for k, peak_index in enumerate(peak_dict[key][1])])
                print(key, avg_pos_value)

                ax.axvline(x=avg_pos_data, color='r', ls='--', label=f'{mean}' + r' $\pm$ ' + f'{std}' + r' [$\lambda / 2$]')
            else:
                break  # No higher index peaks available

        locs, labels = plt.xticks()
        xticks = [int(lead_data_class.x_boundless_data[int(index_offset + index)]) for index in locs[0:-1]]
        plt.xticks(locs[0:-1], xticks)
        # plt.xticks(locs[0:-1], np.linspace(start=value_bound[0], stop=value_bound[1], num=len(locs), dtype=int))
        plt.title(f'Focus mode (q={focus[i][0]}, m+n={focus[i][1]}). Transmission relative to fundamental mode')
        plt.ylabel(f'Different Iterations [a.u.]', fontsize=font_size)
        plt.xlabel(f'Cavity length [nm]', fontsize=font_size)
        plt.grid(True)
        plt.legend()


if __name__ == '__main__':
    from src.import_data import FileToMeasData
    from src.peak_identifier import identify_peaks
    from src.peak_relation import get_converted_measurement_data, get_piezo_response
    # Reference files
    file_samp = 'samples_1s_10V_rate1300000.0'

    def get_collections(iter_count: int) -> Iterable[LabeledPeakCollection]:
        response_func = None
        for i in range(iter_count):
            try:
                filename = file_fetch_function(iteration=i)
                measurement_class = FileToMeasData(meas_file=filename, samp_file=file_samp)
                # # Get response function
                # if response_func is None:
                #     response_func = get_piezo_response(meas_class=measurement_class)
                # # Set all subsequent response functions
                # measurement_class.set_voltage_conversion(conversion_function=response_func)

                measurement_class = get_converted_measurement_data(meas_class=measurement_class)
                labeled_collection = LabeledPeakCollection(identify_peaks(meas_data=measurement_class))
                yield labeled_collection
            except FileNotFoundError:
                break

    iter_count = 30
    plot_pinned_focus_top(collection_iterator=get_collections(iter_count=iter_count), pin=(1, 0), focus=[(1, 2)])
    # plot_pinned_focus_top(collection_iterator=get_collections(iter_count=iter_count), pin=(0, 0), focus=[(1, 2)])

    # plot_pinned_focus_side(collection_iterator=get_collections(iter_count=iter_count), pin=(0, 3))

    plt.show()
