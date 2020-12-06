import numpy as np
from typing import List, Union, Tuple
from src.import_data import SyncMeasData
from src.peak_identifier import PeakCollection, PeakData, identify_peaks


class LabeledPeak(PeakData):
    """PeakData with additional longitudinal and transverse mode labels"""

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

    @property
    def get_avg_x(self) -> float:
        """Returns average data point x-location in cluster."""
        return np.mean([peak.get_x for peak in self._list])

    @property
    def get_avg_y(self) -> float:
        """Returns average data point y-location in cluster."""
        return np.mean([peak.get_y for peak in self._list])

    @property
    def get_value_slice(self) -> Tuple[float, float]:
        """Returns standard deviation data point x-location in cluster."""
        margin = .005
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
        self._mode_clusters = self._set_labeled_peaks(optical_mode_collection)  # Single calculation for performance
        super().__init__(flatten_clusters(data=self._mode_clusters))

    def _set_labeled_peaks(self, optical_mode_collection: PeakCollection) -> List[LabeledPeakCluster]:
        mode_clusters = self._get_clusters(peak_list=optical_mode_collection)  # Construct clusters
        mode_clusters = sorted(mode_clusters, key=lambda x: -x.get_avg_x)  # Sort clusters from small to large cavity
        mode_clusters_height_sorted = sorted(mode_clusters, key=lambda x: -x.get_avg_y)
        height_detection_limit = mode_clusters_height_sorted[0].get_avg_y * 0.3  # TODO: Hardcoded 8 suspected long modes
        ordered_clusters = []  # first index corresponds to long_mode, second index to trans_mode
        # Order based on average y
        for cluster in mode_clusters:
            # TODO: specify the number of long modes you expect (in this case 8)
            if np.max(cluster) >= height_detection_limit:  # > .5:  # TODO: Hardcoded. Should be: ordered_clusters[-1][-1].get_avg_y:
                ordered_clusters.append([cluster])
            elif len(ordered_clusters) == 0:
                continue  # Skip junk peaks before first fundamental mode
            else:
                ordered_clusters[-1].append(cluster)
        # Iterate and label
        result = []
        for long_mode_id, cluster_array in enumerate(ordered_clusters):
            for trans_mode_id, cluster in enumerate(cluster_array):
                result.append(LabeledPeakCluster(data=cluster._list, long_mode=long_mode_id, trans_mode=trans_mode_id))
        return result  # [peak_data for peak_data in optical_mode_collection]

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
            next_mode = self.get_labeled_clusters(long_mode=long_mode + 1, trans_mode=None)

            def estimate_q(sequence: List[LabeledPeakCluster]) -> float:
                if len(sequence) < 2:
                    raise ValueError(f'Not enough modes found to accurately estimate the position of ground mode q')
                distances = [sequence[i+1].get_avg_x - sequence[i].get_avg_x for i in range(len(sequence) - 1)]
                return sequence[0].get_avg_x - np.mean(distances)

            value_slice = (estimate_q(cluster_array), estimate_q(next_mode))
            if value_slice[0] > value_slice[1]:  # Swap data_slice order to maintain (low, high) bound structure
                value_slice = (value_slice[1], value_slice[0])
            return cluster_array, value_slice
        else:  # Sequence specific transverse mode
            cluster_array = self.get_labeled_clusters(long_mode=long_mode, trans_mode=trans_mode)
            return cluster_array, cluster_array[0].get_value_slice

    @staticmethod
    def _get_clusters(peak_list: Union[List[PeakData], PeakCollection]) -> List[PeakCluster]:
        """Separates data points into clusters based on their mutual distance."""
        cluster_data = [peak.get_x for peak in peak_list]
        # Calculate mutual peak distances
        distances = [cluster_data[i + 1] - cluster_data[i] for i in range(len(cluster_data) - 1)]
        mean = np.mean(distances)
        std = np.std(distances)
        # Detect statistical outliers
        outliers = [i for i, distance in enumerate(distances) if
                    abs(distance) > mean + .6 * std]  # TODO: Hardcoded cluster separation
        # Construct cluster splitting
        split_indices = (0,) + tuple(data + 1 for data in outliers) + (len(cluster_data) + 1,)
        return [PeakCluster(peak_list[start: end]) for start, end in zip(split_indices, split_indices[1:])]

    @property
    def get_clusters(self) -> List[LabeledPeakCluster]:
        """Returns a list of pre-calculated mode clusters"""
        return self._mode_clusters

    @property
    def get_clusters_avg_x(self) -> List[float]:
        """Returns average data point x-location in each cluster."""
        return [peak_cluster.get_avg_x for peak_cluster in self.get_clusters]

    @property
    def get_clusters_avg_y(self) -> List[float]:
        """Returns average data point y-location in each cluster."""
        return [peak_cluster.get_avg_y for peak_cluster in self.get_clusters]


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


def get_value_to_data_slice(data_class: SyncMeasData, value_slice: Tuple[float, float]) -> Tuple[int, int]:
    """Transforms arbitrary value data_slice into index specific data_slice"""

    def find_nearest(array: np.ndarray, value: float) -> int: # Assumes data is sorted
        index = np.searchsorted(array, value, side="left")
        if index > 0 and (index == len(array) or abs(value - array[index-1]) < abs(value - array[index])):
            return index-1
        else:
            return index

    return find_nearest(array=data_class.x_boundless_data, value=value_slice[0]), find_nearest(array=data_class.x_boundless_data, value=value_slice[1])


def get_slice_range(data_slice: Tuple[int, int]) -> int:
    return abs(data_slice[0] - data_slice[1])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.plot_npy import prepare_measurement_plot, plot_specific_peaks, plot_peak_collection, plot_class
    from src.peak_identifier import identify_noise_ceiling

    # Construct measurement class
    ax, measurement_class = prepare_measurement_plot('transrefl_hene_1s_10V_PMT5_rate1300000.0_pol000')
    ax2, measurement_class2 = prepare_measurement_plot('transrefl_hene_1s_10V_PMT5_rate1300000.0itteration1')
    # Optional, define data_slice
    # data_slice = (1050000, 1150000)
    # measurement_class.slicer = data_slice  # Zooms in on relevant data part
    # measurement_class2.slicer = data_slice

    # Collect peaks
    peak_collection_itt0 = LabeledPeakCollection(identify_peaks(measurement_class))
    print(len(peak_collection_itt0))
    # peak_collection_itt1 = LabeledPeakCollection(identify_peaks(measurement_class2))
    # Get correlation offset
    # offset_info = get_cluster_offset(peak_collection_itt0, peak_collection_itt1)

    # Plot measurement
    ax = plot_class(axis=ax, measurement_class=measurement_class)
    # ax2 = plot_class(axis=ax2, measurement_class=measurement_class2)

    # Plot peak collection
    # ax = plot_peak_collection(axis=ax, data=peak_collection_itt0)  # All peaks

    cluster_array, value_slice = peak_collection_itt0.get_mode_sequence(long_mode=0)
    data_slice = get_value_to_data_slice(measurement_class, value_slice)

    ax = plot_peak_collection(axis=ax, data=flatten_clusters(data=peak_collection_itt0.get_clusters))

    ax.axvline(x=value_slice[0], color='r', alpha=1)
    ax.axvline(x=value_slice[1], color='g', alpha=1)

    measurement_class.slicer = data_slice
    ax2 = plot_class(axis=ax2, measurement_class=measurement_class)
    for cluster in cluster_array:
        ax2 = plot_peak_collection(axis=ax2, data=cluster)

    ax = plot_specific_peaks(axis=ax, data=peak_collection_itt0, long_mode=None, trans_mode=0)

    # Draw noise h-lines
    noise_ceiling = identify_noise_ceiling(measurement_class)
    ax.axhline(y=noise_ceiling, color='r')

    # Show
    ax2.legend()
    plt.show()
