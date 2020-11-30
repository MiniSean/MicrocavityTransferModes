import matplotlib.pyplot as plt
from src.plot_npy import get_standard_axis, plot_class, plot_peak_collection
from src.import_data import SyncMeasData
from src.peak_relation import LabeledPeakCollection, get_slice, flatten_clusters


def plot_isolated_long_mode(axis: plt.axes, data_class: SyncMeasData, collection: LabeledPeakCollection, long_mode: int) -> plt.axis:
    # Plot array
    cluster_array, value_slice = collection.get_mode_sequence(long_mode=long_mode)
    data_slice = get_slice(data_class=data_class, value_slice=value_slice)
    # Update measurement data indices (slice)
    data_class.slicer = data_slice
    # Update peak indices
    peak_array = []
    for peak in flatten_clusters(data=cluster_array):
        if peak.update_index() is not None:
            peak_array.append(peak)

    # Plot data (slice)
    axis = plot_class(axis=axis, measurement_class=data_class)
    # Plot peak array
    axis = plot_peak_collection(axis=axis, data=peak_array)
    return get_standard_axis(axis=axis)


if __name__ == '__main__':
    from src.peak_identifier import identify_peaks

    # Reference files
    file_samp = 'samples_1s_10V_rate1300000.0'

    # Measurement files
    filename_base = 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration'
    filenames = [filename_base + str(i) for i in range(5)]
    meas_iterations = [SyncMeasData(meas_file=file_meas, samp_file=file_samp, scan_file=None) for file_meas in filenames]
    identified_peaks = [identify_peaks(meas_data=data) for data in meas_iterations]
    labeled_peaks = [LabeledPeakCollection(optical_mode_collection=collection) for collection in identified_peaks]

    # Test
    fig, ax = plt.subplots()
    index = 0
    plot_isolated_long_mode(axis=ax, data_class=meas_iterations[index], collection=labeled_peaks[index], long_mode=0)
    plt.show()
