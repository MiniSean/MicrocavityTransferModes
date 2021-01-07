import numpy as np
from typing import Union, Iterable, Dict, Any, Tuple
from src.import_data import import_npy
from src.peak_relation import LabeledPeakCollection


def extract_mode_separation(collection: LabeledPeakCollection) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, float, float]]:
    """
    :param collection:
    :return: ((t_l, t_t), (dt_l, dt_t)) : (#, diff_squared, diff_abs)
    """
    result = {}

    cluster_array = collection.get_clusters
    for i, cluster_t in enumerate(cluster_array):
        t_l = cluster_t.get_longitudinal_mode_id
        t_t = cluster_t.get_transverse_mode_id
        for j, cluster_dt in enumerate(cluster_array[i+1:]):
            dt_l = cluster_dt.get_longitudinal_mode_id
            dt_t = cluster_dt.get_transverse_mode_id
            abs_difference = np.abs(cluster_dt[-1].get_x - cluster_t[-1].get_x)
            # Memory save storage
            key = ((t_l, t_t), (dt_l, dt_t))
            if key not in result:
                result[key] = (1, abs_difference**2, abs_difference)
            else:
                result[key] = (result[key][0] + 1, result[key][1] + abs_difference**2, result[key][2] + abs_difference)
    return result


def get_allen_variance(collection_iterator: Iterable[LabeledPeakCollection], scan_velocity: float) -> np.ndarray:
    for collection in collection_iterator:
        extract_mode_separation(collection=collection)


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

    allen_variance = get_allen_variance(collection_iterator=get_collections(iter_count=1), scan_velocity=3500)
    # plt.plot(allen_variance)
    # plt.show()
