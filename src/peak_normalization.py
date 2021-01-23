import numpy as np
from typing import Tuple, Union, List
from src.peak_identifier import PeakCollection, PeakData
from src.peak_relation import LabeledPeakCollection, LabeledPeakCluster, LabeledPeak, flatten_clusters


class NormalizedPeak(LabeledPeak):
    """
    LabeledPeak with additional normalized positioning.
    """

    def __new__(cls, labeled_peak: LabeledPeak, **kwargs):
        value = labeled_peak._data.y_data[labeled_peak._raw_index]
        return float.__new__(cls, value)

    def __init__(self, labeled_peak: LabeledPeak, anchor_data: LabeledPeakCollection):
        super().__init__(labeled_peak, labeled_peak.get_longitudinal_mode_id, labeled_peak.get_transverse_mode_id)
        self._anchor_point = self._get_anchor_points(anchor_data=anchor_data)

    def _get_anchor_points(self, anchor_data: LabeledPeakCollection) -> Tuple[Union[LabeledPeakCluster, None], Union[LabeledPeakCluster, None]]:
        """Returns planar-mode clusters based on anchor points."""
        try:
            return anchor_data.q_dict[self.get_longitudinal_mode_id], anchor_data.q_dict[self.get_longitudinal_mode_id + 1]
        except KeyError:
            return None, None

    @property
    def get_norm_x(self) -> Union[float, None]:
        """Returns normalized position (L_qmn - L_q00) / (lambda / 2)."""
        divider = self._anchor_point[1].get_avg_x - self._anchor_point[0].get_avg_x
        if self._anchor_point[0] is None or self._anchor_point[1] is None or divider == 0:
            return None
        else:
            return (self.get_x - self._anchor_point[0].get_avg_x) / divider


class NormalizedPeakCluster(LabeledPeakCluster):
    """
    LabeledPeakCluster with additional normalized positioning.
    """

    def __init__(self, data: LabeledPeakCluster, anchor_data: LabeledPeakCollection):
        super().__init__(data=data, long_mode=data.get_longitudinal_mode_id, trans_mode=data.get_transverse_mode_id)
        self._list = [NormalizedPeak(labeled_peak=labeled_peak, anchor_data=anchor_data) for labeled_peak in self._list]

    @property
    def get_norm_avg_x(self) -> float:
        """Returns normalized average data point x-location in cluster."""
        try:
            return np.mean([peak.get_norm_x for peak in self._list])
        except AttributeError:
            return float('nan')

    @property
    def get_norm_std_x(self) -> float:
        """Returns normalized standard deviation from data point x-location in cluster."""
        try:
            return np.std([peak.get_norm_x for peak in self._list])
        except AttributeError:
            return float('nan')


# Adds additional normalization functionality to the labeled peak collection
class NormalizedPeakCollection(LabeledPeakCollection):
    """
    LabeledPeakCollection with additional normalization functionality to the labeled peak collection.
    """

    def __init__(self, transmission_peak_collection: PeakCollection):
        super().__init__(transmission_peak_collection)  # Constructs q_dict
        # Update internals to represent normalized peak data
        self._mode_clusters = self._set_norm_peaks(self._list)
        self._list = flatten_clusters(data=self._mode_clusters)  # Update internal collection with normalized data

    def _set_norm_peaks(self, optical_mode_collection: Union[List[PeakData], PeakCollection]) -> List[NormalizedPeakCluster]:
        """Returns updated PeakDataClusters by applying normalization."""
        cluster_array = self._set_labeled_clusters(optical_mode_collection)
        result = [NormalizedPeakCluster(data=cluster, anchor_data=self) for cluster in cluster_array]
        return result

    def get_normalized_meas_data_slice(self, union_slice: Union[Tuple[float, float], Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns measurement data class x and y array based on value/data_slice."""
        # Assumes ordered x_array
        x_array, y_array = self.get_measurement_data_slice(union_slice=union_slice)
        min_value = min(x_array)
        rel_max_value = max(x_array) - min_value
        x_array = (x_array - min_value) / rel_max_value  # Normalize array
        y_array = np.asarray([y for _, y in sorted(zip(x_array, y_array), key=lambda pair: self._get_data_class.sort_key(pair[0]))])  # Sort array
        return x_array, y_array

    def get_normalized_mode(self, long_mode: int, trans_mode: Union[int, None]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns measurement data class x and y array based on mode identity."""
        # Full fundamental sequence used for normalization
        sequence_cluster_array, sequence_slice = self.get_mode_sequence(long_mode=long_mode, trans_mode=None)
        full_x_array, full_y_array = self.get_measurement_data_slice(union_slice=sequence_slice)
        min_value = min(full_x_array)
        max_value = max(full_x_array)
        rel_max_value = max_value - min_value
        # Actual modes being displayed
        cluster_array, value_slice = self.get_mode_sequence(long_mode=long_mode, trans_mode=trans_mode)
        # Assumes ordered x_array
        x_array, y_array = self.get_measurement_data_slice(union_slice=value_slice)
        x_array = (x_array - min_value) / rel_max_value  # Normalize array
        y_array = np.asarray([y for _, y in sorted(zip(x_array, y_array), key=lambda pair: self._get_data_class.sort_key(pair[0]))])  # Sort array
        return x_array, y_array

    @property
    def get_clusters(self) -> List[NormalizedPeakCluster]:
        """Returns a list of pre-calculated mode clusters"""
        return self._mode_clusters

