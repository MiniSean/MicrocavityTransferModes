from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from src.plot_npy import get_standard_axis
from src.peak_relation import LabeledPeakCollection, get_converted_measurement_data, get_value_to_data_slice, get_slice_range
from src.peak_normalization import NormalizedPeakCollection
from src.plot_polarized_mode import plot_isolated_long_mode


def get_free_overlap(axis: plt.axes, collection_classes: List[LabeledPeakCollection], long_mode: int, trans_mode: Union[int, None]) -> plt.axes:
    for collection in collection_classes:
        axis = plot_isolated_long_mode(axis=axis, data_class=collection._get_data_class, collection=collection, long_mode=long_mode, trans_mode=trans_mode)
    return get_standard_axis(axis=axis)


def get_focused_overlap(axis: plt.axes, collection_classes: List[NormalizedPeakCollection], long_mode: int, trans_mode: Union[int, None]) -> plt.axes:
    alpha = 1
    for norm_collection in collection_classes:
        x_sample, y_measure = norm_collection.get_normalized_mode(long_mode=long_mode, trans_mode=trans_mode)
        axis.plot(x_sample, y_measure, alpha=alpha)
        alpha = 0.3
    # Set axis
    axis = get_standard_axis(axis=axis)
    axis.set_xlabel('Relative distance between estimated q-modes [a.u.]')  # 'Normalized units [a.u.]')
    return axis


def get_matching_overlap(axis: plt.axes, collection_classes: List[NormalizedPeakCollection], long_mode: int, trans_mode: Union[int, None]) -> plt.axes:
    def get_normalized_cluster_pos(collection: NormalizedPeakCollection) -> float:
        _cluster = collection.get_labeled_clusters(long_mode=long_mode, trans_mode=trans_mode)
        _norm_pos = _cluster[0][-1].get_norm_x  # np.mean([peak.get_norm_x for peak in _cluster[0]])
        return _norm_pos

    reference_norm_pos = get_normalized_cluster_pos(collection=collection_classes[0])
    alpha = 1
    for norm_collection in collection_classes:
        x_sample, y_measure = norm_collection.get_normalized_mode(long_mode=long_mode, trans_mode=trans_mode)
        x_diff = reference_norm_pos - get_normalized_cluster_pos(collection=norm_collection)
        axis.plot(x_sample + x_diff, y_measure, alpha=alpha)
        alpha = 0.3
    # Set axis
    axis = get_standard_axis(axis=axis)
    axis.set_xlabel('Normalized reference units [a.u.]')
    return axis


if __name__ == '__main__':
    from src.import_data import SyncMeasData
    from src.peak_identifier import identify_peaks

    # Reference files
    file_samp = 'samples_0_3s_10V_rate1300000.0'  # 'samples_1s_10V_rate1300000.0'

    # Measurement files
    filenames = [f'transrefl_hene_0_3s_10V_PMT4_rate1300000.0itteration{i}_pol000' for i in range(10)]

    meas_iterations = [get_converted_measurement_data(SyncMeasData(meas_file=file_meas, samp_file=file_samp)) for file_meas in filenames]
    identified_peaks = [identify_peaks(meas_data=data) for data in meas_iterations]
    labeled_peaks = [LabeledPeakCollection(optical_mode_collection=collection) for collection in identified_peaks]
    norm_peaks = [NormalizedPeakCollection(optical_mode_collection=collection) for collection in identified_peaks]

    # TEMP
    long_mode = 0
    trans_mode = 1

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
