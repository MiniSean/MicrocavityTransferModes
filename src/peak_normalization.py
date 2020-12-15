import numpy as np
from typing import Tuple, Union, List
from src.peak_identifier import PeakCollection, PeakData
from src.peak_relation import LabeledPeakCollection, LabeledPeakCluster, LabeledPeak, flatten_clusters, get_value_to_data_slice


class NormalizedPeak(LabeledPeak):
    """LabeledPeak with additional normalized positioning"""

    def __new__(cls, labeled_peak: LabeledPeak, **kwargs):
        value = labeled_peak._data.y_data[labeled_peak._raw_index]
        return float.__new__(cls, value)

    def __init__(self, labeled_peak: LabeledPeak, anchor_data: LabeledPeakCollection):
        super().__init__(labeled_peak, labeled_peak.get_longitudinal_mode_id, labeled_peak.get_transverse_mode_id)
        self._anchor_point = self._get_anchor_points(anchor_data=anchor_data)

    def _get_anchor_points(self, anchor_data: LabeledPeakCollection) -> Tuple[Union[LabeledPeak, None], Union[LabeledPeak, None]]:
        try:
            return anchor_data.q_dict[self.get_longitudinal_mode_id], anchor_data.q_dict[self.get_longitudinal_mode_id + 1]
        except KeyError:
            return None, None

    @property
    def get_norm_x(self) -> Union[float, None]:
        divider = self._anchor_point[1] - self._anchor_point[0]
        if self._anchor_point[0] is None or self._anchor_point[1] is None or divider == 0:
            return None
        else:
            return (self.get_x - self._anchor_point[0].get_x) / (self._anchor_point[1].get_x - self._anchor_point[0].get_x)


# Adds additional normalization functionality to the labeled peak collection
class NormalizedPeakCollection(LabeledPeakCollection):
    """LabeledPeakCollection with additional normalization functionality to the labeled peak collection"""

    def __init__(self, optical_mode_collection: PeakCollection):
        super().__init__(optical_mode_collection)  # Constructs q_dict
        # Update internals to represent normalized peak data
        self._mode_clusters = self._set_norm_peaks(self._list)
        self._list = flatten_clusters(data=self._mode_clusters)  # Update internal collection with normalized data

    def _set_norm_peaks(self, optical_mode_collection: Union[List[PeakData], PeakCollection]) -> List[LabeledPeakCluster]:
        cluster_array = self._set_labeled_clusters(optical_mode_collection)
        for i, cluster in enumerate(cluster_array):
            for j, peak in enumerate(cluster):
                cluster[j] = NormalizedPeak(labeled_peak=peak, anchor_data=self)
        return cluster_array

    def get_normalized_meas_data_slice(self, union_slice: Union[Tuple[float, float], Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        # Assumes ordered x_array
        x_array, y_array = self.get_measurement_data_slice(union_slice=union_slice)
        min_value = min(x_array)
        rel_max_value = max(x_array) - min_value
        x_array = (x_array - min_value) / rel_max_value  # Normalize array
        y_array = np.asarray([y for _, y in sorted(zip(x_array, y_array), key=lambda pair: self._get_data_class.sort_key(pair[0]))])  # Sort array
        # x_array = np.asarray(sorted(x_array, key=lambda x: x))  # Sort array
        return x_array, y_array

    def get_normalized_mode(self, long_mode: int, trans_mode: Union[int, None]) -> Tuple[np.ndarray, np.ndarray]:
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.plot_npy import prepare_measurement_plot
    from src.peak_identifier import identify_peaks
    from src.peak_relation import get_converted_measurement_data
    # Construct measurement class
    ax, measurement_class = prepare_measurement_plot('transrefl_hene_0_3s_10V_PMT4_rate1300000.0itteration0_pol000')
    # Normalized peak collection
    norm_peak_collection = NormalizedPeakCollection(identify_peaks(get_converted_measurement_data(measurement_class)))

    # Test plot
    def peak_inclusion(peak: NormalizedPeak) -> bool:
        return peak.get_norm_x is not None and 0 <= peak.get_norm_x <= 1
    alpha = .5
    nr_sequences = 2
    for i in range(nr_sequences):
        cluster_array, value_slice = norm_peak_collection.get_mode_sequence(long_mode=i)
        # Get normalized measurement
        x_sample, y_measure = norm_peak_collection.get_normalized_meas_data_slice(union_slice=value_slice)
        ax.plot(x_sample, y_measure, alpha=alpha)
        # Get normalized peaks
        peak_array = flatten_clusters(data=cluster_array)
        y = [peak.get_y for peak in peak_array if peak_inclusion(peak)]
        x = [peak.get_norm_x for peak in peak_array if peak_inclusion(peak)]
        ax.plot(x, y, 'o', alpha=alpha)
    ax.set_yscale('log')
    ax.set_title(f'Normalized sequences')
    ax.set_xlabel('Relative distance between estimated q-modes [a.u.]')

    fig, ax2 = plt.subplots()
    for i in range(nr_sequences):
        cluster_array, value_slice = norm_peak_collection.get_mode_sequence(long_mode=i)
        # Get normalized measurement
        x_sample, y_measure = norm_peak_collection.get_measurement_data_slice(union_slice=value_slice)
        ax2.plot(x_sample, y_measure, alpha=alpha)
        # Get normalized peaks
        peak_array = flatten_clusters(data=cluster_array)
        y = [peak.get_y for peak in peak_array if peak_inclusion(peak)]
        x = [peak.get_x for peak in peak_array if peak_inclusion(peak)]
        ax2.plot(x, y, 'o', alpha=alpha)
    ax2.set_yscale('log')
    ax2.set_title(f'Standard sequences')
    ax2.set_xlabel('Sampling Voltage [V]')

    plt.show()

