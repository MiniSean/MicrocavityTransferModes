import numpy as np
from typing import Union, Iterable, Dict, Any, Tuple, List
from src.import_data import import_npy
from src.peak_relation import LabeledPeakCollection
LM_ARRAY = [0, 1, 2, 3, 4, 5, 6]
TM_ARRAY = [0, 1, 2]


def extract_mode_separation(collection: LabeledPeakCollection, pass_dict: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[List[float], List[float]]], scan_velocity: float) -> Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[List[float], List[float]]]:
    """
    :param collection:
    :return: ((t_l, t_t), (dt_l, dt_t)) : (#, diff_squared, diff_abs)
    """
    cluster_array = collection.get_clusters
    lim_cluster_array = []
    for cluster in cluster_array:
        if cluster.get_longitudinal_mode_id not in LM_ARRAY:
            continue
        lim_cluster_array.append(cluster)

    for i, cluster_t in enumerate(lim_cluster_array):
        t_l = cluster_t.get_longitudinal_mode_id
        t_t = cluster_t.get_transverse_mode_id
        for p, peak_t in enumerate(cluster_t):

            for j, cluster_dt in enumerate(lim_cluster_array):  # Only compare to upper left matrix triangle (excluding diagonal)
                if j < i:
                    continue
                dt_l = cluster_dt.get_longitudinal_mode_id
                dt_t = cluster_dt.get_transverse_mode_id
                # peak_range = cluster_dt[p+1:] if i == j else cluster_dt  # Only compare to upper left matrix triangle (excluding diagonal)
                for q, peak_dt in enumerate(cluster_dt):
                    if i == j and q <= p:
                        continue
                    if t_t not in TM_ARRAY or dt_t not in TM_ARRAY:
                        continue

                    # Collect data
                    abs_difference = peak_dt.get_x - peak_t.get_x
                    # abs_difference = np.abs(cluster_dt.get_avg_x - cluster_t.get_avg_x)
                    # Memory save storage
                    key = ((t_l, t_t, p), (dt_l, dt_t, q))
                    if key not in pass_dict:
                        pass_dict[key] = ([abs_difference], [abs(abs_difference)/scan_velocity])
                    else:
                        pass_dict[key][0].append(abs_difference)
                        pass_dict[key][1].append(abs_difference/scan_velocity)
    return pass_dict


def extract_mode_separation_array(collection: LabeledPeakCollection, pass_array: List[Tuple[float, float]], scan_velocity: float) -> List[Tuple[float, float]]:
    cluster_array = collection.get_clusters
    for i, cluster_t in enumerate(cluster_array):
        for p, peak_t in enumerate(cluster_t):
            for j, cluster_dt in enumerate(cluster_array[i+1:]):  # Only compare to upper left matrix triangle (excluding diagonal)
                peak_range = cluster_dt[p+1:] if i == j else cluster_dt  # Only compare to upper left matrix triangle (excluding diagonal)
                for q, peak_dt in enumerate(peak_range):

                    # Collect data
                    abs_difference = np.abs(peak_dt.get_x - peak_t.get_x)
                    pass_array.append((abs_difference, abs_difference/scan_velocity))
    return pass_array


def get_outlier_based_variance(ref_array: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    # Sort based on delta time
    _ref_array = [(sqrt_diff, dt) for sqrt_diff, dt in sorted(ref_array, key=lambda pair: pair[1])]

    # Cluster data points together
    sorted_distances = [sqrt_diff for sqrt_diff, _ in _ref_array]
    sorted_delta_time = [time for _, time in _ref_array]

    time_distances = [(sorted_delta_time[i + 1] - sorted_delta_time[i]) for i in range(len(sorted_delta_time) - 1)]
    mean = np.mean(time_distances)
    std = np.std(time_distances)
    cut_off = mean + 2 * std  # TODO: Hardcoded time interval separation
    # Detect statistical outliers
    outlier_indices = LabeledPeakCollection._get_outlier_indices(values=time_distances, cut_off=cut_off)
    # Construct cluster splitting
    split_indices = (0,) + tuple(data + 1 for data in outlier_indices) + (len(time_distances) + 1,)
    y_array = [np.var(sorted_distances[start: end]) for start, end in zip(split_indices, split_indices[1:])]
    x_array = [np.mean(sorted_delta_time[start: end]) for start, end in zip(split_indices, split_indices[1:])]
    return x_array, y_array


def get_time_interval_variance(ref_array: List[Tuple[float, float]], steps: int) -> Tuple[List[float], List[float]]:
    t_step = int(len(ref_array) / steps)
    split_indices = tuple(i * t_step for i in range(steps)) + (len(ref_array) + 1,)

    # Sort based on delta time
    _ref_array = [(sqrt_diff, dt) for sqrt_diff, dt in sorted(ref_array, key=lambda pair: pair[1])]

    # Cluster data points together
    sorted_distances = [sqrt_diff for sqrt_diff, _ in _ref_array]
    sorted_delta_time = [time for _, time in _ref_array]
    y_array = [np.var(sorted_distances[start: end]) for start, end in zip(split_indices, split_indices[1:])]
    x_array = [np.mean(sorted_delta_time[start: end]) for start, end in zip(split_indices, split_indices[1:])]
    return x_array, y_array


def get_allan_variance(collection_iterator: Iterable[LabeledPeakCollection], scan_velocity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates Allan variance of transmission spectrum"""
    if scan_velocity == 0:
        raise ValueError('Scan velocity needs to be a finite positive number')
    reference_dict = {}
    reference_array = []

    # Collect data
    for collection in collection_iterator:
        reference_dict = extract_mode_separation(collection=collection, pass_dict=reference_dict, scan_velocity=scan_velocity)
        # reference_array = extract_mode_separation_array(collection=collection, pass_array=reference_array, scan_velocity=scan_velocity)

    # Average data
    y_array = []
    x_array = []
    for abs_diff_array, delta_time_array in reference_dict.values():
        if len(abs_diff_array) <= 1:
            continue
        y_var = np.var(abs_diff_array)
        x_var = np.mean(delta_time_array)
        if y_var > 200:  # Outlier
            continue

        y_array.append(y_var)
        x_array.append(x_var)
    # # TEMP
    # plt.plot(x_array, y_array, '.')
    # # Special FSR
    # y_array_0X = []
    # x_array_0X = []
    # y_array_1X = []
    # x_array_1X = []
    # for key in reference_dict.keys():
    #     ((t_l, t_t, p), (dt_l, dt_t, q)) = key
    #     if t_l == 0 and dt_l != 0:
    #         y_array_0X.append(np.var(reference_dict[key][0]))
    #         x_array_0X.append(np.mean(reference_dict[key][1]))
    #     if t_l == 1 and dt_l != 1:
    #         y_array_1X.append(np.var(reference_dict[key][0]))
    #         x_array_1X.append(np.mean(reference_dict[key][1]))
    #
    # plt.plot(x_array_0X, y_array_0X, '.', label=f'1st FSR')
    # plt.plot(x_array_1X, y_array_1X, '.', label=f'2nd FSR')
    # plt.legend()

    return np.asarray(x_array), np.asarray(y_array)


def file_fetch_function(iteration: int) -> Union[str, FileNotFoundError]:
    filename = 'transrefl_hene_1s_10V_PMT4_rate1300000.0itteration{}'.format(iteration)
    try:
        import_npy(filename=filename)
    except FileNotFoundError:
        raise FileNotFoundError(f'File does not exist in pre-defined directory')
    return filename


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.import_data import FileToMeasData
    from src.peak_identifier import identify_peaks
    from src.peak_relation import get_converted_measurement_data
    # Reference files
    file_samp = 'samples_1s_10V_rate1300000.0'

    def get_collections(iter_count: int) -> Iterable[LabeledPeakCollection]:
        for i in range(iter_count):
            try:
                filename = file_fetch_function(iteration=i)
                measurement_class = FileToMeasData(meas_file=filename, samp_file=file_samp)
                measurement_class = get_converted_measurement_data(meas_class=measurement_class)
                labeled_collection = LabeledPeakCollection(identify_peaks(meas_data=measurement_class))
                yield labeled_collection
            except FileNotFoundError:
                break

    iter_count = 30
    scan_speed = 3500
    _x, allan_variance_y = get_allan_variance(collection_iterator=get_collections(iter_count=iter_count), scan_velocity=scan_speed)
    plt.plot(_x, allan_variance_y, '.')

    # Define font
    font_size = 22
    plt.rcParams.update({'font.size': font_size})

    plt.title(f'Allan variance. FSR: {LM_ARRAY[0]}-{LM_ARRAY[-1]}, N: {TM_ARRAY[0]}-{TM_ARRAY[-1]}. ({iter_count} iterations)')
    plt.ylabel(f'Variance' + r' $\langle [x(t + \delta t) - x(t)]^2 \rangle$ $[nm^2]$', fontsize=font_size)
    plt.xlabel(f'Time step' + r' $\delta t[s]$', fontsize=font_size)
    plt.yscale('log')
    # plt.xscale('log')
    # plt.ylim([0, 200000])
    plt.grid(True)

    plt.show()
