import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm  # For displaying for-loop process to console
from typing import Iterable, List, Tuple, Dict
from src.allan_variance_analysis import file_fetch_function
from src.peak_relation import LabeledPeakCollection, get_value_to_data_slice, find_nearest_index, LabeledPeakCluster, SAMPLE_WAVELENGTH


def pad_to_dense(nested_array: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Returns padded array and shape"""
    max_width = np.max([len(i) for i in nested_array])

    result = np.zeros((len(nested_array), max_width))  # Pre-load zero matrix
    for i, row in enumerate(nested_array):
        result[i, :len(row)] += row
    return result, (max_width, len(result))


def pad_to_pinned(nested_array: List[np.ndarray], slice_array: List[Tuple[int, int]], pin_array: List[int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Returns padded array and shape"""
    return pad_to_dense(nested_array=nested_array)


def pad_slice(original: Tuple[int, int], additional: Tuple[int, int]) -> Tuple[int, int]:
    """Returns the Union slice of both original and additional slice"""
    return min(original[0], additional[0]), max(original[1], additional[1])


def store_peak_data(dictionary: Dict[Tuple[int, int, int], List[float]], cluster_array: List[LabeledPeakCluster]) -> Dict[Tuple[int, int, int], List[float]]:
    """Returns dictionary in format: Dict[Tuple[long, trans, index], Tuple[count, position-sum]]"""
    for cluster in cluster_array:
        for k, peak in enumerate(cluster):
            key = (cluster.get_longitudinal_mode_id, cluster.get_transverse_mode_id, k)
            if key in dictionary:
                dictionary[key].append(peak.get_x)
            else:
                dictionary[key] = [peak.get_x]
    return dictionary


def plot_pinned_focus(collection_iterator: Iterable[LabeledPeakCollection], pin: Tuple[int, int], focus: List[Tuple[int, int]]):  #  -> plt.axes
    """Create color plot which pins on single mode and displays one or more reference modes"""

    # Prepare parameters
    iter_count = 0
    largest_array_len = 0
    lead_data_class = None
    peak_dict = {}  # Dict[Tuple(long, trans, index), Tuple(count, position sum)]
    total_value_slice_array = []
    total_data_slice_array = []
    trans_array = []
    value_focus_array = []
    data_class_array = []
    for i, collection in tqdm(enumerate(collection_iterator), desc=f'Process collections'):
        iter_count = i + 1  # Temp
        # Pin value
        pin_cluster_array, pin_value_slice = collection.get_mode_sequence(long_mode=pin[0], trans_mode=pin[1])
        # Store peak data
        peak_dict = store_peak_data(dictionary=peak_dict, cluster_array=pin_cluster_array)

        # Get corresponding data_slice (index slice) and pin index
        data_class = collection._get_data_class  # Access SyncMeasData class corresponding to peak data
        # Focus value
        for j, sub_focus in enumerate(focus):
            foc_cluster_array, foc_value_slice = collection.get_mode_sequence(long_mode=sub_focus[0], trans_mode=sub_focus[1])
            # Store peak data
            peak_dict = store_peak_data(dictionary=peak_dict, cluster_array=foc_cluster_array)

            # Store value slice
            print(sub_focus, foc_value_slice)
            if len(value_focus_array) <= j:
                value_focus_array.insert(j, foc_value_slice)
            else:
                value_focus_array[j] = pad_slice(original=value_focus_array[j], additional=foc_value_slice)

        # total min-max value slice based on pin and focus slices
        total_value_slice = pin_value_slice
        for bound in value_focus_array:
            total_value_slice = pad_slice(original=total_value_slice, additional=bound)
        # Value to data slice
        total_data_slice = get_value_to_data_slice(data_class=data_class, value_slice=total_value_slice)

        total_value_slice_array.append(total_value_slice)  # Store value slice
        total_data_slice_array.append(total_data_slice)  # Store data slice
        # data_pin_array.append(data_pin)  # Store data pin
        trans_array.append(np.asarray(collection.get_measurement_data_slice(union_slice=total_data_slice)[1]))  # Store y-data
        # Store data class of largest array len
        data_class_array.append(data_class)

    # Find data class corresponding to average length

    value_slice_length_array = [value_slice[1] - value_slice[0] for value_slice in total_value_slice_array]
    data_class_index = find_nearest_index(array=value_slice_length_array, value=np.max(value_slice_length_array))
    print(data_class_index)
    lead_data_class = data_class_array[data_class_index]

    # Format data
    lead_index = trans_array.index(max(trans_array, key=len))
    trans_array, array_shape = pad_to_dense(nested_array=trans_array)
    trans_array = np.transpose(trans_array)  # Transpose

    # Data index to distance [nm]
    # Averaged
    # avg_value_slice = (np.mean([value_slice[0] for value_slice in total_value_slice_array]),
    #                    np.mean([value_slice[1] for value_slice in total_value_slice_array]))
    # std_value_slice = (np.std([value_slice[0] for value_slice in total_value_slice_array]),
    #                    np.std([value_slice[1] for value_slice in total_value_slice_array]))
    avg_value_slice = (lead_data_class.x_boundless_data[total_data_slice_array[data_class_index][0]],
                       lead_data_class.x_boundless_data[total_data_slice_array[data_class_index][1]])
    # lead_value_slice = (np.mean([lead_data_class.x_boundless_data[data_slice[0]] for data_slice in total_data_slice_array]),
    #                     np.mean([lead_data_class.x_boundless_data[data_slice[1]] for data_slice in total_data_slice_array]))

    # Define font
    font_size = 22
    plt.rcParams.update({'font.size': font_size})

    # Total slice
    x, y = np.mgrid[0:array_shape[0], 0:array_shape[1]]
    plt.pcolormesh(x, y, trans_array, norm=colors.LogNorm())

    locs, labels = plt.xticks()
    plt.xticks(locs[0:-1], np.linspace(start=avg_value_slice[0], stop=avg_value_slice[1], num=len(locs), dtype=int))
    plt.title(f'Pinned plot over {iter_count} sample iteration(s)')
    plt.ylabel(f'Different Iterations [a.u.]', fontsize=font_size)
    #  with {round(std_value_slice[0], 2)}, {round(std_value_slice[1], 2)} ' + r'$\sigma$' + '
    plt.xlabel(f'Cavity length (based on average) [nm]', fontsize=font_size)
    plt.grid(True)

    index_offset = total_data_slice_array[lead_index][0]
    for i, bound in enumerate(value_focus_array):
        data_bound = get_value_to_data_slice(data_class=lead_data_class, value_slice=bound)  # Sample to index
        fig, ax = plt.subplots()
        array_bound = (data_bound[0] - index_offset, data_bound[1] - index_offset)

        focus_trans_array = trans_array[array_bound[0]:array_bound[1]]  # After transpose
        focus_array_shape = focus_trans_array.shape
        # Focus slice
        x, y = np.mgrid[0:focus_array_shape[0], 0:focus_array_shape[1]]
        ax.pcolormesh(x, y, focus_trans_array, norm=colors.LogNorm())

        # Plot peak average position
        rev_key = (pin[0], pin[1], 0)
        rev_pos_value = np.mean(peak_dict[rev_key])  # Sum / Count [nm]
        long_mode, trans_mode = focus[i]
        for j in range(10):
            key = (long_mode, trans_mode, j)
            if key in peak_dict:
                avg_pos_value = np.mean(peak_dict[key])  # Sum / Count [nm]
                std_pos_value = np.std(peak_dict[key])
                avg_pos_data = find_nearest_index(array=lead_data_class.x_boundless_data, value=avg_pos_value) - (index_offset + array_bound[0])
                # (avg_pos_value - rev_pos_value)/(SAMPLE_WAVELENGTH/2)
                ax.axvline(x=avg_pos_data, color='r', ls='--', label=f'{round(avg_pos_value, 5)}' + r' $\pm$ ' + f'{round(std_pos_value, 2)}' + r' [$\lambda / 2$]')
            else:
                break  # No higher index peaks available

        locs, labels = plt.xticks()
        focus_value_slice = (lead_data_class.x_boundless_data[data_bound[0]], lead_data_class.x_boundless_data[data_bound[1]])
        plt.xticks(locs[0:-1], np.linspace(start=focus_value_slice[0], stop=focus_value_slice[1], num=len(locs), dtype=int))
        plt.title(f'Focus mode (q={focus[i][0]}, m+n={focus[i][1]}). Transmission relative to fundamental mode')
        plt.ylabel(f'Different Iterations [a.u.]', fontsize=font_size)
        plt.xlabel(f'Cavity length (based on largest measurement) [nm]', fontsize=font_size)
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

    iter_count = 300
    plot_pinned_focus(collection_iterator=get_collections(iter_count=iter_count), pin=(0, 0), focus=[(0, 1), (0, 2)])

    plt.show()
