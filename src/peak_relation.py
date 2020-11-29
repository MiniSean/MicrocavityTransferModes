import numpy as np
from src.plot_npy import plot_peak_collection
from src.peak_identifier import PeakCollection, identify_peaks


def get_cluster_offset(base_observer: PeakCollection, target_observer: PeakCollection) -> float:
    cluster_mean_base = base_observer.get_cluster_averages()
    cluster_mean_target = target_observer.get_cluster_averages()
    return np.mean([base - target for base, target in zip(cluster_mean_base, cluster_mean_target)])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.plot_npy import prepare_measurement_plot, plot_class
    # Construct measurement class
    ax, measurement_class = prepare_measurement_plot('transrefl_hene_1s_10V_PMT5_rate1300000.0itteration0')
    ax2, measurement_class2 = prepare_measurement_plot('transrefl_hene_1s_10V_PMT5_rate1300000.0itteration1')
    # Optional, define slice
    slice = (1050000, 1150000)
    measurement_class.slicer = slice  # Zooms in on relevant data part
    measurement_class2.slicer = slice

    # Collect peaks
    peak_collection_itt0 = identify_peaks(measurement_class)
    peak_collection_itt1 = identify_peaks(measurement_class2)
    # Get correlation offset
    offset_info = get_cluster_offset(peak_collection_itt0, peak_collection_itt1)
    clusters = peak_collection_itt0.get_clusters()

    # Plot peak collection
    ax = plot_peak_collection(axis=ax, data=peak_collection_itt0)
    ax = plot_peak_collection(axis=ax, data=peak_collection_itt1)

    for i, cluster_mean in enumerate(peak_collection_itt0.get_cluster_averages()):
        ax.axvline(x=cluster_mean, color='r', alpha=1)
    for i, cluster_mean in enumerate(peak_collection_itt1.get_cluster_averages()):
        ax.axvline(x=cluster_mean, color='g', alpha=1)

    # Show
    plt.show()