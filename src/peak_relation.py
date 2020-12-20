import numpy as np
from typing import List, Union, Tuple, Dict
from src.import_data import SyncMeasData
from src.peak_identifier import PeakCollection, PeakData, identify_peaks
SAMPLE_WAVELENGTH = 633


class LabeledPeak(PeakData):
    """PeakData with additional longitudinal and transverse mode labels"""

    def __new__(cls, peak: PeakData, **kwargs):
        value = peak._data.y_data[peak._raw_index]
        return float.__new__(cls, value)

    def __init__(self, peak: PeakData, long_mode: int, trans_mode: int):
        super().__init__(peak._data, peak._raw_index)
        self._long_mode = long_mode
        self._trans_mode = trans_mode

    @property
    def get_longitudinal_mode_id(self) -> int:
        return self._long_mode

    @property
    def get_transverse_mode_id(self) -> int:
        return self._trans_mode


class PeakCluster:
    def __init__(self, data: List[PeakData]):
        self._list = data

    def __iter__(self):
        return self._list.__iter__()

    def __len__(self):
        return self._list.__len__()

    def __getitem__(self, item):
        return self._list.__getitem__(item)

    def __setitem__(self, key, value):
        return self._list.__setitem__(key, value)

    @property
    def get_avg_x(self) -> float:
        """Returns average data point x-location in cluster."""
        return np.mean([peak.get_x for peak in self._list])

    @property
    def get_avg_y(self) -> float:
        """Returns average data point y-location in cluster."""
        return np.mean([peak.get_y for peak in self._list])

    @property
    def get_std_x(self) -> float:
        """Returns standard deviation from data point x-location in cluster."""
        return np.std([peak.get_x for peak in self._list])

    @property
    def get_std_y(self) -> float:
        """Returns standard deviation from data point y-location in cluster."""
        return np.std([peak.get_y for peak in self._list])

    @property
    def get_max_y(self) -> float:
        """Returns the maximum y-valued peak within the cluster"""
        return sorted(self._list, key=lambda y: y)[-1].get_y

    @property
    def get_value_slice(self) -> Tuple[float, float]:
        """Returns standard deviation data point x-location in cluster."""
        margin = 0.5 * self.get_std_x + 0.001 * (max(self._list[0]._data.x_boundless_data) - min(self._list[0]._data.x_boundless_data))
        return min([peak.get_x for peak in self._list]) - margin, max([peak.get_x for peak in self._list]) + margin  # np.std([peak.get_x for peak in self._list])


class LabeledPeakCluster(PeakCluster):
    def __init__(self, data: List[PeakData], long_mode: int, trans_mode: int):
        super().__init__([LabeledPeak(peak=peak, long_mode=long_mode, trans_mode=trans_mode) for peak in data])
        self._long_mode = long_mode
        self._trans_mode = trans_mode

    @property
    def get_longitudinal_mode_id(self) -> int:
        return self._long_mode

    @property
    def get_transverse_mode_id(self) -> int:
        return self._trans_mode


class LabeledPeakCollection(PeakCollection):
    def __init__(self, optical_mode_collection: PeakCollection):
        self._mode_clusters = self._set_labeled_clusters(optical_mode_collection)  # Pre sample conversion
        super().__init__(flatten_clusters(data=self._mode_clusters))
        self.q_dict = self._set_q_dict(cluster_array=self._mode_clusters)

    def _set_labeled_clusters(self, optical_mode_collection: Union[List[PeakData], PeakCollection]) -> List[LabeledPeakCluster]:
        mode_clusters = self._get_clusters(peak_list=optical_mode_collection)  # Construct clusters
        sort_key = mode_clusters[0][0]._data.sort_key  # Cluster sorting key (Small-to-Large [nm] or Large-to-Small [V])
        mode_clusters = sorted(mode_clusters, key=lambda x: sort_key(x.get_avg_x))  # Sort clusters from small to large cavity

        # Use first and second order difference to handle transverse mode cluster overlap between two main modes
        # mode_y_distances = [(mode_clusters[i].get_avg_y - mode_clusters[i-1].get_avg_y) for i in range(len(mode_clusters)-1)]
        # mode_y_distances_2nd = [(mode_clusters[i].get_avg_y - mode_clusters[i-2].get_avg_y) for i in range(len(mode_clusters)-1)]

        mode_high_distances = [(np.log(mode_clusters[i].get_max_y) - np.log(mode_clusters[i-1].get_max_y)) for i in range(len(mode_clusters)-1)]
        mode_high_distances_2nd = [(np.log(mode_clusters[i].get_max_y) - np.log(mode_clusters[i-2].get_max_y)) for i in range(len(mode_clusters)-1)]

        mean = np.mean(mode_high_distances)
        std = np.std(mode_high_distances)
        cut_off = (mean + 0.75 * std)  # TODO: Hardcoded cut-off value for fundamental peak outliers
        fundamental_indices = [i for i in range(len(mode_high_distances)) if mode_high_distances[i] > cut_off and mode_high_distances_2nd[i] > cut_off]
        overlap_indices = [i-1 for i in range(len(mode_high_distances)) if mode_high_distances[i] > cut_off and mode_high_distances_2nd[i] < cut_off]

        ordered_clusters = []  # first index corresponds to long_mode, second index to trans_mode
        # Order based on average y
        for i, cluster in enumerate(mode_clusters):
            if i in fundamental_indices:  # Identify fundamental peaks
                ordered_clusters.append([cluster])
            elif len(ordered_clusters) == 0:
                continue  # Skip junk peaks before first fundamental mode
            elif i in overlap_indices:  # Identify overlap transverse modes
                if len(ordered_clusters) >= 2:
                    ordered_clusters[-2].append(cluster)
                else:
                    continue
            else:
                ordered_clusters[-1].append(cluster)
        # Iterate and label
        result = []
        for long_mode_id, cluster_array in enumerate(ordered_clusters):
            for trans_mode_id, cluster in enumerate(cluster_array):
                result.append(LabeledPeakCluster(data=cluster._list, long_mode=long_mode_id, trans_mode=trans_mode_id))
        return result  # [peak_data for peak_data in optical_mode_collection]

    def _set_q_dict(self, cluster_array: List[LabeledPeakCluster]) -> Dict[int, Union[LabeledPeak, None]]:
        result = {}
        data_class = self._get_data_class
        for i in range(cluster_array[-1].get_longitudinal_mode_id):
            try:  # find q mode estimate until not enough data points are available
                value = self.get_q_estimate(long_mode=i)
            except ValueError:  # Add None identifier for last q mode
                result[i] = None
                return result
            data_index = find_nearest_index(array=data_class.x_boundless_data, value=value)
            peak = LabeledPeak(PeakData(data=data_class, index=data_index), long_mode=i, trans_mode=-1)
            result[i] = peak
        return result

    def get_labeled_clusters(self, long_mode: Union[int, None], trans_mode: Union[int, None]) -> List[LabeledPeakCluster]:
        result = []

        def filter_func(cluster: LabeledPeakCluster):  # Filter function for specific mode label
            return (True if long_mode is None else cluster.get_longitudinal_mode_id == long_mode) and \
                   (True if trans_mode is None else cluster.get_transverse_mode_id == trans_mode)

        for labeled_peak in self._mode_clusters:
            if filter_func(cluster=labeled_peak):
                result.append(labeled_peak)
        return result

    def get_labeled_peaks(self, long_mode: Union[int, None], trans_mode: Union[int, None]) -> List[LabeledPeak]:
        """The referred labeled clusters contain labeled peaks by definition"""
        return flatten_clusters(data=self.get_labeled_clusters(long_mode=long_mode, trans_mode=trans_mode))

    def get_mode_sequence(self, long_mode: int, trans_mode: Union[int, None] = None) -> Tuple[List[LabeledPeakCluster], Tuple[float, float]]:
        """
        Determines the data_slice bounds in which the requested longitudinal mode exists compared to its successor.
        :param long_mode: Requested longitudinal mode number
        :param trans_mode: Requested transverse mode number
        :return: Tuple( All clusters corresponding to longitudinal mode, data_slice boundary in arbitrary cavity space units)
        """
        if trans_mode is None:  # Sequence entire longitudinal mode slice
            # Create sequence data_slice
            cluster_array = self.get_labeled_clusters(long_mode=long_mode, trans_mode=None)
            value_slice = (self.get_q_estimate(long_mode=long_mode), self.get_q_estimate(long_mode=long_mode + 1))
            if value_slice[0] > value_slice[1]:  # Swap data_slice order to maintain (low, high) bound structure
                value_slice = (value_slice[1], value_slice[0])
            return cluster_array, value_slice
        else:  # Sequence specific transverse mode
            cluster_array = self.get_labeled_clusters(long_mode=long_mode, trans_mode=trans_mode)
            return cluster_array, cluster_array[0].get_value_slice

    def get_q_estimate(self, long_mode: int) -> float:
        sequence = self.get_labeled_clusters(long_mode=long_mode, trans_mode=None)
        if len(sequence) < 2:
            raise ValueError(f'Not enough modes found to accurately estimate the position of ground mode q')
        distances = [sequence[i + 1].get_avg_x - sequence[i].get_avg_x for i in range(len(sequence) - 1)]
        # distances = [sequence[1].get_avg_x - sequence[0].get_avg_x]  # Only relative to first transverse mode
        return sequence[0].get_avg_x - np.mean(distances)

    def get_measurement_data_slice(self, union_slice: Union[Tuple[float, float], Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns sample (x) and measurement (y) array data from requested data/value slice."""
        # Work with data slice
        data_class = self._get_data_class
        data_slice = union_slice
        if isinstance(data_slice[0], np.float64) and isinstance(data_slice[1], np.float64):
            data_slice = get_value_to_data_slice(data_class=data_class, value_slice=union_slice)

        # Collect relevant x and y array from data class
        return data_class.x_boundless_data[data_slice[0]:data_slice[1]], data_class.y_boundless_data[data_slice[0]:data_slice[1]]

    @staticmethod
    def _get_clusters(peak_list: Union[List[PeakData], PeakCollection]) -> List[PeakCluster]:
        """Separates data points into clusters based on their mutual distance."""
        cluster_data = [peak.get_x for peak in peak_list]
        # Calculate mutual peak distances
        distances = [(cluster_data[i + 1] - cluster_data[i]) for i in range(len(cluster_data) - 1)]
        mean = np.mean(distances)
        std = np.std(distances)
        # print(max(distances) - min(distances))  # 0.2497
        # print(.24 * (mean + 2 * std))  # 0.0409
        cut_off = 0.24 * (mean + 2 * std)  # 0.03  # mean + .3 * std  # TODO: Hardcoded cluster separation
        # Detect statistical outliers
        outlier_indices = LabeledPeakCollection._get_outlier_indices(values=distances, cut_off=cut_off)
        # Construct cluster splitting
        split_indices = (0,) + tuple(data + 1 for data in outlier_indices) + (len(cluster_data) + 1,)
        return [PeakCluster(peak_list[start: end]) for start, end in zip(split_indices, split_indices[1:])]

    @staticmethod
    def _get_outlier_indices(values: List[float], cut_off: float) -> List[int]:
        return [i for i, value in enumerate(values) if abs(value) > cut_off]

    @property
    def _get_data_class(self) -> SyncMeasData:
        return self._mode_clusters[0][0]._data

    @property
    def get_clusters(self) -> List[LabeledPeakCluster]:
        """Returns a list of pre-calculated mode clusters"""
        return self._mode_clusters

    @property
    def get_q_clusters(self) -> List[LabeledPeakCluster]:
        """Returns a list of q (m + n = 1) clusters"""
        result = []
        for q_long_id, q_peak in self.q_dict.items():
            result.append(LabeledPeakCluster(data=[q_peak], long_mode=q_long_id, trans_mode=-1))
        return result

    @property
    def get_clusters_avg_x(self) -> List[float]:
        """Returns average data point x-location in each cluster."""
        return [peak_cluster.get_avg_x for peak_cluster in self.get_clusters]

    @property
    def get_clusters_avg_y(self) -> List[float]:
        """Returns average data point y-location in each cluster."""
        return [peak_cluster.get_avg_y for peak_cluster in self.get_clusters]


def get_converted_measurement_data(meas_class: SyncMeasData) -> SyncMeasData:
    from src.sample_conversion import fit_piezo_response
    initial_labeling = LabeledPeakCollection(identify_peaks(meas_class))
    # Set voltage conversion function
    response_func = fit_piezo_response(cluster_collection=initial_labeling.get_clusters, sample_wavelength=SAMPLE_WAVELENGTH)
    meas_class.set_voltage_conversion(conversion_function=response_func)
    return meas_class


def get_cluster_offset(base_observer: LabeledPeakCollection, target_observer: LabeledPeakCollection) -> float:
    cluster_mean_base = base_observer.get_clusters_avg_x
    cluster_mean_target = target_observer.get_clusters_avg_x
    return np.mean([base - target for base, target in zip(cluster_mean_base, cluster_mean_target)])


def flatten_clusters(data: List[PeakCluster]) -> List[PeakData]:
    result = []
    for cluster in data:
        for peak in cluster:
            result.append(peak)
    return result


def find_nearest_index(array: np.ndarray, value: float) -> int: # Assumes data is sorted
    index = np.searchsorted(array, value, side="left")
    if index > 0 and (index == len(array) or abs(value - array[index-1]) < abs(value - array[index])):
        return index-1
    else:
        return index


def get_value_to_data_slice(data_class: SyncMeasData, value_slice: Tuple[float, float]) -> Tuple[int, int]:
    """Transforms arbitrary value data_slice into index specific data_slice"""
    return find_nearest_index(array=data_class.x_boundless_data, value=value_slice[0]), find_nearest_index(array=data_class.x_boundless_data, value=value_slice[1])


def get_slice_range(data_slice: Tuple[int, int]) -> int:
    return abs(data_slice[0] - data_slice[1])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.plot_npy import prepare_measurement_plot, plot_specific_peaks, plot_peak_collection, plot_class
    from src.peak_identifier import identify_noise_ceiling

    # # Temp
    # from src.structural_analysis import MeasurementAnalysis
    # analysis = MeasurementAnalysis(meas_file='transrefl_hene_0_3s_10V_PMT4_rate1300000.0itteration0_pol000', samp_file='samples_0_3s_10V_rate1300000.0', scan_file=None)
    # print(analysis)

    # Construct measurement class
    ax, measurement_class = prepare_measurement_plot('transrefl_hene_0_3s_10V_PMT4_rate1300000.0itteration0_pol080')
    ax2, measurement_class2 = prepare_measurement_plot('transrefl_hene_0_3s_10V_PMT4_rate1300000.0itteration0_pol000')
    #     # Optional, define data_slice
    # data_slice = (1050000, 1150000)
    # measurement_class.slicer = data_slice  # Zooms in on relevant data part
    # measurement_class2.slicer = data_slice

    # Collect peaks
    measurement_class = get_converted_measurement_data(meas_class=measurement_class)
    labeled_collection = LabeledPeakCollection(identify_peaks(meas_data=measurement_class))
    print(len(labeled_collection))

    # Plot measurement
    ax = plot_class(axis=ax, measurement_class=measurement_class)
    # ax2 = plot_class(axis=ax2, measurement_class=measurement_class2)

    # Plot peak collection
    # ax = plot_peak_collection(axis=ax, data=peak_collection_itt0)  # All peaks

    cluster_array, value_slice = labeled_collection.get_mode_sequence(long_mode=0)
    data_slice = get_value_to_data_slice(measurement_class, value_slice)

    ax = plot_peak_collection(axis=ax, data=flatten_clusters(data=labeled_collection.get_clusters))

    ax.axvline(x=value_slice[0], color='r', alpha=1)
    ax.axvline(x=value_slice[1], color='g', alpha=1)

    measurement_class.slicer = data_slice
    ax2 = plot_class(axis=ax2, measurement_class=measurement_class)
    # cluster_array, value_slice = labeled_collection.get_mode_sequence(long_mode=0, trans_mode=1)
    for cluster in cluster_array:
        ax2 = plot_peak_collection(axis=ax2, data=cluster)

    ax = plot_specific_peaks(axis=ax, data=labeled_collection, long_mode=None, trans_mode=0)

    # Draw noise h-lines
    noise_ceiling = identify_noise_ceiling(measurement_class)
    ax.axhline(y=noise_ceiling, color='r')

    # Show
    # ax2.legend()
    plt.show()
