import numpy as np
from typing import Union, Iterable, Dict, Any, Tuple
from src.import_data import import_npy
from src.peak_relation import LabeledPeakCollection


def extract_mode_separation(collection: LabeledPeakCollection, pass_dict: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[int, float, float]], scan_velocity: float) -> Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[int, float, float]]:
    """
    :param collection:
    :return: ((t_l, t_t), (dt_l, dt_t)) : (#, diff_squared, diff_abs)
    """
    cluster_array = collection.get_clusters
    for i, cluster_t in enumerate(cluster_array):
        t_l = cluster_t.get_longitudinal_mode_id
        t_t = cluster_t.get_transverse_mode_id
        for p, peak_t in enumerate(cluster_t):

            for j, cluster_dt in enumerate(cluster_array[i:]):  # Only compare to upper left matrix triangle (excluding diagonal)
                dt_l = cluster_dt.get_longitudinal_mode_id
                dt_t = cluster_dt.get_transverse_mode_id
                peak_range = cluster_dt[p+1:] if i == j else cluster_dt  # Only compare to upper left matrix triangle (excluding diagonal)
                for q, peak_dt in enumerate(peak_range):

                    # Collect data
                    abs_difference = np.abs(peak_dt.get_x - peak_t.get_x)
                    # Memory save storage
                    key = ((t_l, t_t, p), (dt_l, dt_t, q))
                    if key not in pass_dict:
                        pass_dict[key] = (1, abs_difference**2, abs_difference/scan_velocity)
                    else:
                        pass_dict[key] = (pass_dict[key][0] + 1, pass_dict[key][1] + abs_difference**2, pass_dict[key][2] + abs_difference/scan_velocity)
    return pass_dict


def get_allan_variance(collection_iterator: Iterable[LabeledPeakCollection], scan_velocity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates Allan variance of transmission spectrum"""
    if scan_velocity == 0:
        raise ValueError('Scan velocity needs to be a finite positive number')
    reference_dict = {}

    # Collect data
    for collection in collection_iterator:
        reference_dict = extract_mode_separation(collection=collection, pass_dict=reference_dict, scan_velocity=scan_velocity)

    # Average data
    y_array = []
    x_array = []
    for count, abs_diff, delta_time in reference_dict.values():
        y_array.append(abs_diff / count)
        x_array.append(delta_time / count)

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
            filename = file_fetch_function(iteration=i)
            measurement_class = FileToMeasData(meas_file=filename, samp_file=file_samp)
            measurement_class = get_converted_measurement_data(meas_class=measurement_class)
            labeled_collection = LabeledPeakCollection(identify_peaks(meas_data=measurement_class))
            yield labeled_collection

    iter_count = 10
    scan_speed = 1
    _x, allan_variance_y = get_allan_variance(collection_iterator=get_collections(iter_count=iter_count), scan_velocity=scan_speed)
    plt.plot(_x, allan_variance_y, '.')

    # Define font
    font_size = 24
    plt.rcParams.update({'font.size': font_size})

    plt.title(f'Allan variance over {iter_count} sample iteration(s)')
    plt.ylabel(f'Variance' + r' $\langle [x(t + \delta t) - x(t)]^2 \rangle$', fontsize=font_size)
    plt.xlabel(f'Time step' + r' $\delta t[s]$', fontsize=font_size)
    plt.yscale('log')
    plt.xscale('log')
    # plt.ylim([0, 200000])
    plt.grid(True)

    plt.show()
