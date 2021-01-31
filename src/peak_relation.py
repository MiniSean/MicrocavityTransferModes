import numpy as np
from typing import List, Union, Tuple, Dict, Callable, Optional
from src.import_data import SyncMeasData
from src.peak_identifier import PeakCollection, PeakData, identify_peaks
SAMPLE_WAVELENGTH = 633
# cutoff = mean + c * std
HEIGHT_SEPARATION = 0.65  # [0, 2]: Minimum height value between clusters-max to specify as new fundamental mode
CLUSTER_SEPARATION = 0.1  # [0, 2]: Minimum distance value between peaks within a single cluster


class LabeledPeak(PeakData):
    """
    PeakData with additional longitudinal and transverse mode labels.
    """

    def __new__(cls, peak: PeakData, **kwargs):
        value = peak._data.y_data[peak._raw_index]
        return float.__new__(cls, value)

    def __init__(self, peak: PeakData, long_mode: int, trans_mode: int):
        super().__init__(peak._data, peak._raw_index)
        self._long_mode = long_mode
        self._trans_mode = trans_mode

    @property
    def get_longitudinal_mode_id(self) -> int:
        """Returns longitudinal mode number."""
        return self._long_mode

    @property
    def get_transverse_mode_id(self) -> int:
        """Returns transverse mode number."""
        return self._trans_mode


class PeakCluster:
    """
    Groups PeakData into a single mode cluster
    """

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
        margin = 1  # 0.5 * self.get_std_x + 0.001 * (max(self._list[0]._data.x_boundless_data) - min(self._list[0]._data.x_boundless_data))
        return min([peak.get_x for peak in self._list]) - margin, max([peak.get_x for peak in self._list]) + margin  # np.std([peak.get_x for peak in self._list])


class LabeledPeakCluster(PeakCluster):
    """
    PeakCluster with additional longitudinal and transverse mode labels.
    """

    def __init__(self, data: List[PeakData], long_mode: int, trans_mode: int):
        super().__init__([LabeledPeak(peak=peak, long_mode=long_mode, trans_mode=trans_mode) for peak in data])
        self._long_mode = long_mode
        self._trans_mode = trans_mode

    @property
    def get_longitudinal_mode_id(self) -> int:
        """Returns longitudinal mode number."""
        return self._long_mode

    @property
    def get_transverse_mode_id(self) -> int:
        """Returns transverse mode number."""
        return self._trans_mode


class LabeledPeakCollection(PeakCollection):
    """
    Array collection of LabeledPeak class.
    Contains array of LabeledPeakCluster classes containing the LabeledPeak class.
    Contains dictionary with estimated (m + n = -1) or (planar) mode.
    """

    def __init__(self, transmission_peak_collection: PeakCollection, q_offset: int = 0, custom_height_cutoff: float = HEIGHT_SEPARATION, custom_cluster_seperation: float = CLUSTER_SEPARATION):
        self._q_offset = q_offset
        self._custom_height_cutoff = custom_height_cutoff
        self._custom_cluster_seperation = custom_cluster_seperation
        self._mode_clusters = self._set_labeled_clusters(transmission_peak_collection)  # Pre sample conversion
        super().__init__(flatten_clusters(data=self._mode_clusters))
        self.q_dict = self._set_q_dict(cluster_array=self._mode_clusters)

    def _set_labeled_clusters(self, optical_mode_collection: Union[List[PeakData], PeakCollection]) -> List[LabeledPeakCluster]:
        """
        Orders and labels the provided array of PeakData classes.
        :param optical_mode_collection: Array of PeakData.
        :return: Array of LabeledPeakCluster.
        """
        mode_clusters = self._get_clusters(peak_list=optical_mode_collection)  # Construct clusters
        # Overlap separation
        # for mode in mode_clusters:
        #     peak_height = sorted([peak.get_y for peak in mode], key=lambda x: -x)
        #     peak_height_distances = [peak_height[i] - peak_height[i + 1] for i in range(len(peak_height) - 1)]
        #     if len(peak_height_distances) < 2:
        #         continue
        #
        #     mean = np.mean(peak_height_distances)
        #     std = np.std(peak_height_distances)
        #     cut_off = (mean + 2.5 * std)
        #     outlier_indices = LabeledPeakCollection._get_outlier_indices(values=peak_height_distances, cut_off=cut_off)
        #     if len(outlier_indices) > 0:
        #         print(mode.get_avg_x, len(outlier_indices))
        #         temp.append(mode.get_avg_x)

        sort_key = mode_clusters[0][0]._data.sort_key  # Cluster sorting key (Small-to-Large [nm] or Large-to-Small [V])
        mode_clusters = sorted(mode_clusters, key=lambda x: sort_key(x.get_avg_x))  # Sort clusters from small to large cavity

        mode_high_distances = [(np.log(mode_clusters[i].get_max_y) - np.log(mode_clusters[i-1].get_max_y)) for i in range(len(mode_clusters)-1)]
        mode_high_distances_2nd = [(np.log(mode_clusters[i].get_max_y) - np.log(mode_clusters[i-2].get_max_y)) for i in range(len(mode_clusters)-1)]

        mean = np.mean(mode_high_distances)
        std = np.std(mode_high_distances)
        cut_off = (mean + self._custom_height_cutoff * std)  # TODO: Hardcoded cut-off value for fundamental peak outliers
        fundamental_indices = [i for i in range(len(mode_high_distances)) if mode_high_distances[i] > cut_off and mode_high_distances_2nd[i] > cut_off]
        overlap_indices = [i-1 for i in range(len(mode_high_distances)) if mode_high_distances[i] > cut_off and mode_high_distances_2nd[i] < cut_off]

        ordered_clusters = []  # first index corresponds to long_mode, second index to trans_mode
        # Order based on average y
        for i, cluster in enumerate(mode_clusters):
            if i in fundamental_indices:  # and (len(ordered_clusters) != 0 or abs(cluster.get_avg_x - ordered_clusters[-1][0].get_avg_x) > SAMPLE_WAVELENGTH * .4):  # Identify fundamental peaks
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
                result.append(LabeledPeakCluster(data=cluster._list, long_mode=long_mode_id + self._q_offset, trans_mode=trans_mode_id))
        return result  # [peak_data for peak_data in optical_mode_collection]

    def _set_q_dict(self, cluster_array: List[LabeledPeakCluster]) -> Dict[int, Union[LabeledPeakCluster, None]]:
        """
        Creates a dictionary for estimated m+n=-1 modes.
        These clusters consists of predicted peak positions based on all transverse modes in the relevant longitudinal mode.
        :param cluster_array: All identified clusters within the data.
        :return: Dict[ longitudinal mode, (m+n=-1) peak cluster ].
        """
        result = {}
        data_class = self._get_data_class
        for i in range(cluster_array[-1].get_longitudinal_mode_id):
            try:  # find q mode estimate until not enough data points are available
                pos_array = self.get_q_estimate(long_mode=i)
            except ValueError:  # Not enough data points to determine q mode (Add None identifier for last q mode)
                continue
                # result[i] = None
                # return result
            # Construct labeled peak cluster
            peak_array = []
            for j, pos_value in enumerate(pos_array):
                data_index = find_nearest_index(array=data_class.x_boundless_data, value=pos_value)
                peak = LabeledPeak(PeakData(data=data_class, index=data_index), long_mode=i, trans_mode=-1)
                peak_array.append(peak)
            cluster = LabeledPeakCluster(data=peak_array, long_mode=i, trans_mode=-1)

            result[i] = cluster
        return result

    def get_labeled_clusters(self, long_mode: Union[int, None], trans_mode: Union[int, None]) -> List[LabeledPeakCluster]:
        """
        Returns labeled clusters filtered by longitudinal or transverse mode identity.
        When either long_mode or trans_mode is parsed with None, all the respective mode identities are filtered.
        :param long_mode: Int of longitudinal identity.
        :param trans_mode: Int of transverse identity.
        :return: Array of LabeledPeakClusters corresponding to the filter requirements.
        """
        result = []

        def filter_func(cluster: LabeledPeakCluster):  # Filter function for specific mode label
            return (True if long_mode is None else cluster.get_longitudinal_mode_id == long_mode) and \
                   (True if trans_mode is None else cluster.get_transverse_mode_id == trans_mode)

        for labeled_peak in self._mode_clusters:
            if filter_func(cluster=labeled_peak):
                result.append(labeled_peak)
        return result

    def get_labeled_peaks(self, long_mode: Union[int, None], trans_mode: Union[int, None]) -> List[LabeledPeak]:
        """
        Returns labeled peaks filtered by longitudinal or transverse mode identity.
        When either long_mode or trans_mode is parsed with None, all the respective mode identities are filtered.
        :param long_mode: Int of longitudinal identity.
        :param trans_mode: Int of transverse identity.
        :return: Array of LabeledPeak corresponding to the filter requirements.
        """
        return flatten_clusters(data=self.get_labeled_clusters(long_mode=long_mode, trans_mode=trans_mode))

    def get_mode_sequence(self, long_mode: int, trans_mode: Union[int, None] = None) -> Tuple[List[LabeledPeakCluster], Tuple[float, float]]:
        """
        Determines the data_slice bounds of the FSR corresponding to the longitudinal mode identity.
        :param long_mode: Int of longitudinal identity.
        :param trans_mode: Int of transverse identity.
        :return: Tuple( All clusters corresponding to longitudinal mode, data_slice boundary in arbitrary cavity space units).
        """
        if trans_mode is None:  # Sequence entire longitudinal mode slice
            # Create sequence data_slice
            cluster_array = self.get_labeled_clusters(long_mode=long_mode, trans_mode=None)
            value_slice = (np.mean(self.get_q_estimate(long_mode=long_mode)), np.mean(self.get_q_estimate(long_mode=long_mode + 1)))
            if value_slice[0] > value_slice[1]:  # Swap data_slice order to maintain (low, high) bound structure
                value_slice = (value_slice[1], value_slice[0])
            return cluster_array, value_slice
        else:  # Sequence specific transverse mode
            cluster_array = self.get_labeled_clusters(long_mode=long_mode, trans_mode=trans_mode)
            return cluster_array, cluster_array[0].get_value_slice

    def get_q_estimate(self, long_mode: int) -> List[float]:
        """
        Predicts m+n=-1 mode based on higher order transverse mode locations.
        :param long_mode: Specific longitudinal mode to search for relevant transverse modes.
        :return: List[ positions depending on all relevant transverse modes ].
        """
        sequence = self.get_labeled_clusters(long_mode=long_mode, trans_mode=None)
        if len(sequence) < 2:
            raise ValueError(f'Not enough modes found to accurately estimate the position of ground mode q')
        max_range = len(sequence) - 1
        distances = [sequence[i + 1].get_avg_x - sequence[i].get_avg_x for i in range(max_range)]
        return [sequence[0].get_avg_x - dist for dist in distances]

    def get_measurement_data_slice(self, union_slice: Union[Tuple[float, float], Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns sample (x) and measurement (y) array data from requested data/value slice."""
        # Work with data slice
        data_class = self._get_data_class
        data_slice = union_slice
        if isinstance(data_slice[0], np.float64) and isinstance(data_slice[1], np.float64):
            data_slice = get_value_to_data_slice(data_class=data_class, value_slice=union_slice)

        # Collect relevant x and y array from data class
        return data_class.x_boundless_data[data_slice[0]:data_slice[1]], data_class.y_boundless_data[data_slice[0]:data_slice[1]]

    # @staticmethod
    def _get_clusters(self, peak_list: Union[List[PeakData], PeakCollection]) -> List[PeakCluster]:
        """Separates data points into clusters based on their mutual distance."""
        cluster_data = [peak.get_x for peak in peak_list]
        # Calculate mutual peak distances
        distances = [(cluster_data[i + 1] - cluster_data[i]) for i in range(len(cluster_data) - 1)]
        mean = np.mean(distances)
        std = np.std(distances)
        cut_off = mean + self._custom_cluster_seperation * std  # TODO: Hardcoded cluster separation
        # Detect statistical outliers
        outlier_indices = LabeledPeakCollection._get_outlier_indices(values=distances, cut_off=cut_off)
        # Construct cluster splitting
        split_indices = (0,) + tuple(data + 1 for data in outlier_indices) + (len(cluster_data) + 1,)
        return [PeakCluster(peak_list[start: end]) for start, end in zip(split_indices, split_indices[1:])]

    @staticmethod
    def _get_outlier_indices(values: List[float], cut_off: float) -> List[int]:
        """Filters all values bellow cut_off threshold."""
        return [i for i, value in enumerate(values) if abs(value) > cut_off]

    @property
    def _get_data_class(self) -> SyncMeasData:
        """
        Returns measurement data class of first PeakData in first PeakCluster.
        Which is the same as all other PeakData inside the collection.
        """
        return self._mode_clusters[0][0]._data

    @property
    def get_clusters(self) -> List[LabeledPeakCluster]:
        """Returns a list of pre-calculated mode clusters."""
        return self._mode_clusters

    @property
    def get_q_clusters(self) -> List[LabeledPeakCluster]:
        """Returns a list of q (m + n = -1) clusters."""
        result = []
        for q_long_id, q_cluster in self.q_dict.items():
            if q_cluster is None:
                raise ValueError(f'(m + n = -1) mode not identified. None found.')
            result.append(q_cluster)
        return result

    @property
    def get_min_q_id(self) -> int:
        return min(self.q_dict.keys())

    @property
    def get_max_q_id(self) -> int:
        return max(self.q_dict.keys())

    @property
    def get_clusters_avg_x(self) -> List[float]:
        """Returns average data point x-location in each cluster."""
        return [peak_cluster.get_avg_x for peak_cluster in self.get_clusters]

    @property
    def get_clusters_avg_y(self) -> List[float]:
        """Returns average data point y-location in each cluster."""
        return [peak_cluster.get_avg_y for peak_cluster in self.get_clusters]


def get_piezo_response(meas_class: Union[SyncMeasData, LabeledPeakCollection], **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns the fitted piezo-voltage to cavity-length response.
    **kwargs:
        - q_offset: (LabeledPeakCollection) determinse the starting fundamental mode
    """
    from src.sample_conversion import fit_piezo_response
    if isinstance(meas_class, SyncMeasData):
        initial_labeling = LabeledPeakCollection(identify_peaks(meas_class), **kwargs)
    else:
        initial_labeling = meas_class
    # Set voltage conversion function
    return fit_piezo_response(cluster_collection=initial_labeling.get_q_clusters, sample_wavelength=SAMPLE_WAVELENGTH)


def get_converted_measurement_data(meas_class: Union[SyncMeasData, LabeledPeakCollection], **kwargs) -> SyncMeasData:
    """
    Returns updated meas_class with converted sample (x-axis) data.
    **kwargs:
        - q_offset: (get_piezo_response) determinse the starting fundamental mode
    """
    # Set voltage conversion function
    response_func = get_piezo_response(meas_class=meas_class, **kwargs)
    if isinstance(meas_class, SyncMeasData):
        result_meas_class = meas_class
    else:
        result_meas_class = meas_class._get_data_class
    result_meas_class.set_voltage_conversion(conversion_function=response_func)
    return result_meas_class


def flatten_clusters(data: List[PeakCluster]) -> List[PeakData]:
    """Returns a flattened array of PeakData from list of PeakClusters."""
    result = []
    for cluster in data:
        for peak in cluster:
            result.append(peak)
    return result


def find_nearest_index(array: np.ndarray, value: float) -> int:  # Assumes data is sorted
    """Returns nearest index of closest array-value to value."""
    index = np.searchsorted(array, value, side="left")
    if index > 0 and (index == len(array) or abs(value - array[index-1]) < abs(value - array[index])):
        return index-1
    else:
        return index


def get_value_to_data_slice(data_class: SyncMeasData, value_slice: Tuple[float, float]) -> Tuple[int, int]:
    """Transforms arbitrary value_slice into index specific data_slice"""
    return find_nearest_index(array=data_class.x_boundless_data, value=value_slice[0]), find_nearest_index(array=data_class.x_boundless_data, value=value_slice[1])


def get_slice_range(data_slice: Tuple[int, int]) -> int:
    """Returns length of data_slice."""
    return abs(data_slice[0] - data_slice[1])
