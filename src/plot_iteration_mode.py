from typing import Union, List, Tuple, Iterator, Callable, Any
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For displaying for-loop process to console
from src.plot_npy import get_standard_axis
from src.peak_relation import LabeledPeakCollection, LabeledPeak, get_converted_measurement_data, get_value_to_data_slice, get_slice_range, flatten_clusters, find_nearest_index
from src.peak_normalization import NormalizedPeakCollection
from src.plot_polarized_mode import plot_isolated_long_mode
FIRST_ALPHA = 1.0
SECOND_ALPHA = 0.5


def get_free_overlap(axis: plt.axes, collection_classes: List[LabeledPeakCollection], long_mode: int, trans_mode: Union[int, None]) -> plt.axes:
    alpha = SECOND_ALPHA
    for collection in collection_classes:
        axis = plot_isolated_long_mode(axis=axis, data_class=collection._get_data_class, collection=collection, long_mode=long_mode, trans_mode=trans_mode, alpha=alpha)
    return get_standard_axis(axis=axis)


def get_focused_overlap(axis: plt.axes, collection_classes: List[NormalizedPeakCollection], long_mode: int, trans_mode: Union[int, None]) -> plt.axes:
    for i, norm_collection in enumerate(collection_classes):
        alpha = SECOND_ALPHA if i == len(collection_classes) - 1 else FIRST_ALPHA
        x_sample, y_measure = norm_collection.get_normalized_mode(long_mode=long_mode, trans_mode=trans_mode)
        axis.plot(x_sample, y_measure, alpha=alpha)
    # Set axis
    axis = get_standard_axis(axis=axis)
    axis.set_xlabel('Relative distance between estimated q-modes')  # 'Normalized units [a.u.]')
    return axis


def get_matching_overlap(axis: plt.axes, collection_classes: List[NormalizedPeakCollection], long_mode: int, trans_mode: Union[int, None]) -> plt.axes:
    def get_normalized_cluster_pos(collection: NormalizedPeakCollection) -> float:
        _cluster = collection.get_labeled_clusters(long_mode=long_mode, trans_mode=trans_mode)
        _norm_pos = _cluster[0][0].get_norm_x  # np.mean([peak.get_norm_x for peak in _cluster[0]])
        return _norm_pos

    reference_norm_pos = get_normalized_cluster_pos(collection=collection_classes[0])
    for i, norm_collection in enumerate(collection_classes):
        alpha = SECOND_ALPHA if i == len(collection_classes) - 1 else FIRST_ALPHA
        x_sample, y_measure = norm_collection.get_normalized_mode(long_mode=long_mode, trans_mode=trans_mode)
        x_diff = reference_norm_pos - get_normalized_cluster_pos(collection=norm_collection)
        axis.plot(x_sample + x_diff, y_measure, alpha=alpha)
    # Set axis
    axis = get_standard_axis(axis=axis)
    axis.set_xlabel('Relative distance between estimated q-modes')  # 'Relative units' + r'$\times 2/\lambda$'
    return axis


def most_frequent(array: List[float]):
    return max(set(array), key=array.count)


def max_frequent(array: List[float]):
    return max(array)


def get_peak_most_frequent_peaks(collection_classes: List[NormalizedPeakCollection], long_mode: int, trans_mode: Union[int, None], force_number: Union[int, None] = None) -> List[List[LabeledPeak]]:
    peak_clusters = []
    for norm_collection in collection_classes:
        cluster_array = norm_collection.get_labeled_clusters(long_mode=long_mode, trans_mode=trans_mode)
        peak_array = flatten_clusters(cluster_array)
        peak_clusters.append(peak_array)

    max_peaks = int(most_frequent([len(cluster) for cluster in peak_clusters]))  # Most frequently found number of peaks
    if force_number is not None:  # Temporary force
        max_peaks = force_number
    peak_clusters = [cluster for cluster in peak_clusters if len(cluster) == max_peaks]
    return peak_clusters  # Trust me I know for certain these are labeled peaks (Since they originate from a normalized collection)


def get_peak_height(collection_classes: List[NormalizedPeakCollection], long_mode: int, trans_mode: Union[int, None], **kwargs) -> Iterator[Tuple[float, float]]:
    peak_clusters = get_peak_most_frequent_peaks(collection_classes=collection_classes, long_mode=long_mode, trans_mode=trans_mode, **kwargs)
    height_array = list(map(list, zip(*peak_clusters)))
    mean_height = [float(np.mean(height_group)) for height_group in height_array]
    std_height = [float(np.std(height_group)) for height_group in height_array]
    return zip(mean_height, std_height)


def get_peak_differences(collection_classes: List[NormalizedPeakCollection], long_mode: int, trans_mode: Union[int, None], **kwargs) -> Iterator[Tuple[float, float]]:
    peak_clusters = get_peak_most_frequent_peaks(collection_classes=collection_classes, long_mode=long_mode, trans_mode=trans_mode, **kwargs)
    # transposed = list(map(list, zip(*peak_clusters)))
    diff_array = []
    # [abs(positions[i + 1] - positions[i]) for positions in transposed for i in range(len(positions) - 1)]
    for peaks in peak_clusters:
        differences = []
        for i in range(len(peaks) - 1):
            differences.append(abs(peaks[i + 1].get_x - peaks[i].get_x))
        diff_array.append(differences)
    diff_array = list(map(list, zip(*diff_array)))
    mean_diff = [float(np.mean(diff_group)) for diff_group in diff_array]
    std_diff = [float(np.std(diff_group)) for diff_group in diff_array]
    return zip(mean_diff, std_diff)


def get_polarized_comparison(filename_func: Callable[[int, int], Union[str, FileNotFoundError]], sample_file: str, long_mode: int, trans_mode: Union[int, None]):
    warnings.filterwarnings(action='once')
    height_data = []
    diff_data = []
    # Collect data
    polarization_iterator = range(0, 19)
    for polarization in tqdm(polarization_iterator, desc=f'Collecting data over polarization sweep'):
        _filenames = []
        for iteration in range(0, 10):
            try:
                _filename = filename_func(iteration, polarization)
            except FileNotFoundError:
                break  # Breaks out of the inner for-loop
            # Successfully retrieved file name
            _filenames.append(_filename)
        # Time for processing into normalized collection
        _meas_iterations = [get_converted_measurement_data(FileToMeasData(meas_file=file_meas, samp_file=sample_file)) for file_meas in _filenames]  # get_converted_measurement_data
        _identified_peaks = [identify_peaks(meas_data=data) for data in _meas_iterations]
        # Temp solution
        # _norm_peaks = [NormalizedPeakCollection(optical_mode_collection=collection) for collection in _identified_peaks]
        _norm_peaks = []
        for collection in _identified_peaks:
            try:
                normalaized_data = NormalizedPeakCollection(transmission_peak_collection=collection)
            except IndexError:  # Sneaky catch for improper normalization (self._get_data_class)
                print(f'Skipped data')
                continue
            _norm_peaks.append(normalaized_data)

        # Process peak data
        # expected_number_of_peaks = 4  # Temp
        _mean_std_height_array = list(get_peak_height(collection_classes=_norm_peaks, long_mode=long_mode, trans_mode=trans_mode))
        _mean_std_diff_array = list(get_peak_differences(collection_classes=_norm_peaks, long_mode=long_mode, trans_mode=trans_mode))
        for i, (mean, std) in enumerate(_mean_std_height_array):
            print(f'\nPeak height {i}: mean = {round(mean, 4)}[Transmission], std = {round(std, 4)}[Transmission]')
        for i, (mean, std) in enumerate(_mean_std_diff_array):
            print(f'Peak difference {i}-{i + 1}: mean = {round(mean, 4)}[nm], std = {round(std, 4)}[nm]')
        # Collect data
        height_data.append(_mean_std_height_array)
        diff_data.append(_mean_std_diff_array)

    def get_match_index(value_array: List[float]) -> Callable[[float], List[int]]:
        _look_up = []
        for i in range(len(value_array)):
            for j in range(1, 2):  # len(value_array) - i + 1
                _look_up.append((np.sum(np.nan_to_num(value_array[i:(i + j)])), list(range(i, i + j))))

        def match_func(value_input: float) -> List[int]:
            look_up_array = np.array([value for value, indices in _look_up])
            index = min(range(len(look_up_array)), key=lambda k: abs(look_up_array[k]-value_input))  # find_nearest_index(array=look_up_array, value=value_input)
            # print(f'Compare {value_input} to:\n{_look_up}\nFinds index {index}, corresponding to {_look_up[index]}')
            return _look_up[index][1]
        return match_func

    def get_iterator_on_max_elements(data: List[Any], correct_len: int) -> List[int]:
        _iterator = list(range(len(data)))
        for i in _iterator:
            if len(data[i]) == correct_len:
                start = _iterator[i:]
                end = _iterator[0:i]
                _iterator = start
                _iterator.extend(end)
                break
        return _iterator

    def estimate_missing_height_data(height_data: List[List[Tuple[float, float]]], diff_data: List[List[Tuple[float, float]]]) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
        """
        Handles undetected peaks throughout data.
        Expects data format:
        List[
            Polarization(0):
            List[
                Peak(0):
                Tuple[
                    height_mean,
                    height_std
                    ]
                Peak(1):
                ...
                ]
            Polarization(1):
            ...
            ]

        The number of peaks per polarization (averaged over multiple iterations) can vary from polarization to polarization.
        Either through peak degeneracy or undetectability.
        Solution approach:
        - establish the max frequency number of peaks
        - use peak-difference data to establish missing peak indices
        """
        # Step 1
        max_peak_freq = int(max_frequent([len(polar_group) for polar_group in height_data]))  # Max number of consistent peaks found
        max_diff_freq = int(max_frequent([len(polar_group) for polar_group in diff_data]))  # Max number of consistent peaks differences found
        # Step 2
        missing_diff_data = []
        # Define data iterator to correct missing data
        diff_iterator = get_iterator_on_max_elements(data=diff_data, correct_len=max_diff_freq)  # list(range(len(diff_data)))

        for i, iter_index in enumerate(diff_iterator):

            if len(diff_data[iter_index]) != max_diff_freq:
                if len(diff_data[diff_iterator[i-1]]) == max_diff_freq:

                    # Normal execution logic
                    corresponding_index_func = get_match_index(value_array=[mean for mean, std in diff_data[diff_iterator[i-1]]])
                    for j, (curr_mean, curr_std) in enumerate(diff_data[iter_index]):
                        index_array = corresponding_index_func(curr_mean)  # Retrieves the reference mean-indices
                        for k in range(j, index_array[0]):
                            diff_data[iter_index].insert(k, (np.nan, np.nan))
                            missing_diff_data.append((iter_index, k))
                        # if len(index_array) > 1:  # One or both bounding peaks are missing
                        #     for k in index_array[1:]:
                        #         diff_data[i].insert(k, (np.nan, np.nan))
                        #         missing_diff_data.append((i, k))
                else:
                    # Problem
                    # raise NotImplementedError?
                    continue

        print(missing_diff_data)
        for (pol_index, list_index) in missing_diff_data:
            height_data[pol_index].insert(list_index, (0, np.nan))

        return height_data, diff_data

    height_data, diff_data = estimate_missing_height_data(height_data=height_data, diff_data=diff_data)

    # Display data
    x = [i * 10 for i in polarization_iterator]
    height_array_mean_std = [list(map(list, zip(*mean_std_list))) for mean_std_list in height_data]  # Polar[ Peaks[] ]
    y_height = list(map(list, zip(*[mean for (mean, std) in height_array_mean_std])))
    y_height_err = list(map(list, zip(*[std for (mean, std) in height_array_mean_std])))
    diff_array_mean_std = [list(map(list, zip(*mean_std_list))) for mean_std_list in diff_data]  # Polar[ Peaks[] ]
    y_diff = list(map(list, zip(*[mean for (mean, std) in diff_array_mean_std])))
    y_diff_err = list(map(list, zip(*[std for (mean, std) in diff_array_mean_std])))
    # Define plot
    fig, (ax0, ax1) = plt.subplots(2, 1)
    for i in range(len(y_height)):
        ax0.errorbar(x=x, y=y_height[i], yerr=y_height_err[i], fmt='', label=f'Peak ({i})')
    for i in range(len(y_diff)):
        ax1.errorbar(x=x, y=y_diff[i], yerr=y_diff_err[i], fmt='', label=f'Peak ({i})-to-({i+1})')
    # Set labels
    fig.suptitle(f'Peak relations focused on (q={long_mode}, m+n={trans_mode})\n(peak labeling: min-max resonance distance)')

    ax0.set_xlabel('Polarization [Degrees]')
    ax0.set_ylabel('Transmission [a.u.]')
    # ax0.set_yscale('log')
    ax0.grid(True)
    ax0.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax1.set_xlabel('Polarization [Degrees]')
    ax1.set_ylabel('Peak Distance [nm]')
    # ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')


if __name__ == '__main__':
    from src.import_data import FileToMeasData, import_npy
    from src.peak_identifier import identify_peaks

    # Reference files
    file_samp = 'samples_1s_10V_rate1300000.0'  # 'samples_1s_10V_rate1300000.0'

    def file_fetch_function(iteration: int, polarization: int) -> Union[str, FileNotFoundError]:
        # _filename = 'transrefl_hene_0_3s_10V_PMT4_rate1300000.0itteration{}_pol{:0=2d}0'.format(iteration, polarization)
        _filename = 'transrefl_hene_1s_10V_PMT4_rate1300000.0itteration{}'.format(iteration)
        try:
            import_npy(filename=_filename)
        except FileNotFoundError:
            raise FileNotFoundError(f'File does not exist in pre-defined directory')
        return _filename

    long_mode = 0
    trans_mode = 1
    # get_polarized_comparison(filename_func=file_fetch_function, sample_file=file_samp, long_mode=long_mode, trans_mode=trans_mode)

    # Measurement files
    # f'transrefl_hene_0_3s_10V_PMT4_rate1300000.0itteration{i}_pol000'
    max_iter = 300
    last_max_count = 10
    filenames = [file_fetch_function(iteration=i, polarization=0) for i in range(max_iter - last_max_count, max_iter)]

    meas_iterations = [get_converted_measurement_data(FileToMeasData(meas_file=file_meas, samp_file=file_samp)) for file_meas in filenames]
    identified_peaks = [identify_peaks(meas_data=data) for data in meas_iterations]
    labeled_peaks = [LabeledPeakCollection(transmission_peak_collection=collection) for collection in identified_peaks]
    norm_peaks = [NormalizedPeakCollection(transmission_peak_collection=collection) for collection in identified_peaks]

    mean_std_diff_array = get_peak_differences(collection_classes=norm_peaks, long_mode=long_mode, trans_mode=trans_mode)
    for i, (mean, std) in enumerate(mean_std_diff_array):
        print(f'Peak difference {i}-{i+1}: mean = {round(mean, 4)}[nm], std = {round(std, 4)}[nm]')
    mean_std_height_array = get_peak_height(collection_classes=norm_peaks, long_mode=long_mode, trans_mode=trans_mode)
    for i, (mean, std) in enumerate(mean_std_height_array):
        print(f'Peak height {i}: mean = {round(mean, 4)}[Transmission], std = {round(std, 4)}[Transmission]')

    fig, ax = plt.subplots()
    ax = get_free_overlap(axis=ax, collection_classes=labeled_peaks, long_mode=long_mode, trans_mode=trans_mode)
    ax.set_title(f'Free overlap (q* = {long_mode}, m+n = {trans_mode})')
    fig2, ax2 = plt.subplots()
    ax2 = get_focused_overlap(axis=ax2, collection_classes=norm_peaks, long_mode=long_mode, trans_mode=trans_mode)
    ax2.set_title(f'Focused overlap (q* = {long_mode}, m+n = {trans_mode})')
    fig3, ax3 = plt.subplots()
    ax3 = get_matching_overlap(axis=ax3, collection_classes=norm_peaks, long_mode=long_mode, trans_mode=trans_mode)
    ax3.set_title(f'Matched overlap (q* = {long_mode}, m+n = {trans_mode})')
    plt.show()
