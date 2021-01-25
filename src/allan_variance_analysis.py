import numpy as np
from typing import Union, Iterable, Dict, Any, Tuple, List
from src.import_data import import_npy
from src.peak_relation import LabeledPeakCollection, find_nearest_index
LM_ARRAY = [2, 3, 4]
TM_ARRAY = [0, 1, 2]


def extract_mode_separation(collection: LabeledPeakCollection, pass_dict: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[List[float], List[float]]], scan_velocity: float) -> Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[List[float], List[float]]]:
    """
    :param collection:
    :return: ((t_l, t_t), (dt_l, dt_t)) : (#, diff_squared, diff_abs)
    """
    cluster_array = collection.get_clusters
    lim_cluster_array = [cluster for cluster in cluster_array if cluster.get_longitudinal_mode_id in LM_ARRAY and cluster.get_transverse_mode_id in TM_ARRAY]
    # for cluster in cluster_array:
    #     if cluster.get_longitudinal_mode_id not in LM_ARRAY:
    #         continue
    #     lim_cluster_array.append(cluster)

    for i, cluster_t in enumerate(lim_cluster_array):
        t_l = cluster_t.get_longitudinal_mode_id
        t_t = cluster_t.get_transverse_mode_id
        # for p, peak_t in enumerate(cluster_t):

        for j, cluster_dt in enumerate(lim_cluster_array):  # Only compare to upper left matrix triangle (excluding diagonal)
            if j < i:
                continue
            dt_l = cluster_dt.get_longitudinal_mode_id
            dt_t = cluster_dt.get_transverse_mode_id
            # peak_range = cluster_dt[p+1:] if i == j else cluster_dt  # Only compare to upper left matrix triangle (excluding diagonal)
            # for q, peak_dt in enumerate(cluster_dt):
            #     if i == j and q <= p:
            #         continue
            #     if t_t not in TM_ARRAY or dt_t not in TM_ARRAY:
            #         continue

            # Collect data
            # abs_difference = peak_dt.get_x - peak_t.get_x
            abs_difference = abs(cluster_dt.get_avg_x - cluster_t.get_avg_x)
            p, q = 0, 0
            # Memory save storage
            key = ((t_l, t_t, p), (dt_l, dt_t, q))
            # key = (dt_l - t_l, dt_t - t_t, q - p)
            if key not in pass_dict:
                pass_dict[key] = ([abs_difference], [abs_difference/scan_velocity])
            else:
                pass_dict[key][0].append(abs_difference)
                pass_dict[key][1].append(abs_difference/scan_velocity)
    return pass_dict


# Deprecated
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


# Deprecated
def get_outlier_based_variance(ref_array: Iterable[Tuple[List[float], List[float]]]) -> Tuple[List[float], List[float]]:
    # Sort based on delta time
    _ref_array = [(sqrt_diff[i], dt[i]) for sqrt_diff, dt in sorted(ref_array, key=lambda pair: pair[1]) for i in range(len(dt))]

    # Cluster data points together
    sorted_distances = [sqrt_diff for sqrt_diff, _ in _ref_array]
    sorted_delta_time = [time for _, time in _ref_array]

    time_distances = [(sorted_delta_time[i + 1] - sorted_delta_time[i]) for i in range(len(sorted_delta_time) - 1)]
    mean = np.mean(time_distances)
    std = np.std(time_distances)
    cut_off = mean + 1.5 * std  # TODO: Hardcoded time interval separation
    # Detect statistical outliers
    outlier_indices = LabeledPeakCollection._get_outlier_indices(values=time_distances, cut_off=cut_off)
    # Construct cluster splitting
    split_indices = (0,) + tuple(data + 1 for data in outlier_indices) + (len(time_distances) + 1,)
    y_array = [np.var(sorted_distances[start: end]) for start, end in zip(split_indices, split_indices[1:])]
    x_array = [np.mean(sorted_delta_time[start: end]) for start, end in zip(split_indices, split_indices[1:])]
    return x_array, y_array


# Deprecated
def get_time_interval_variance(ref_array: Iterable[Tuple[List[float], List[float]]], steps: int) -> Tuple[List[float], List[float]]:
    # Sort based on delta time
    _ref_array = [(sqrt_diff[i], dt[i]) for sqrt_diff, dt in sorted(ref_array, key=lambda pair: pair[1]) for i in range(len(dt))]
    sorted_distances = [sqrt_diff for sqrt_diff, _ in _ref_array]
    sorted_delta_time = [time for _, time in _ref_array]

    t_step = (_ref_array[-1][1] - _ref_array[0][1]) / steps
    split_times = tuple(find_nearest_index(sorted_delta_time, i * t_step) for i in range(steps + 1))

    # Sort based on delta time
    _ref_array = [(sqrt_diff, dt) for sqrt_diff, dt in sorted(ref_array, key=lambda pair: pair[1])]

    # Cluster data points together
    y_array = [np.var(sorted_distances[start: end]) for start, end in zip(split_times, split_times[1:])]
    x_array = [np.mean(sorted_delta_time[start: end]) for start, end in zip(split_times, split_times[1:])]
    return x_array, y_array


def get_allan_variance(collection_iterator: Iterable[LabeledPeakCollection], scan_velocity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates Allan variance of transmission spectrum"""
    if scan_velocity == 0:
        raise ValueError('Scan velocity needs to be a finite positive number')
    reference_dict = {}

    # Collect data
    for collection in collection_iterator:
        reference_dict = extract_mode_separation(collection=collection, pass_dict=reference_dict, scan_velocity=scan_velocity)

    # # Order dictionary
    # ordered_dict = {}
    # for ((t_l, t_t, p), (dt_l, dt_t, q)), (abs_diff_array, delta_time_array) in reference_dict.items():
    #     key = (dt_l - t_l, dt_t - t_t)
    #     if key not in ordered_dict:
    #         ordered_dict[key] = (abs_diff_array, delta_time_array)
    #     else:
    #         ordered_dict[key][0].extend(abs_diff_array)
    #         ordered_dict[key][1].extend(delta_time_array)


    # Sorted
    # x_array, y_array = get_time_interval_variance(ref_array=reference_dict.values(), steps=14)

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

    # # Average data
    # y_array = []
    # x_array = []
    # for abs_diff_array, delta_time_array in reference_dict.values():
    #     y_array.extend(abs_diff_array)
    #     x_array.extend(delta_time_array)

    return np.asarray(x_array), np.asarray(y_array)
