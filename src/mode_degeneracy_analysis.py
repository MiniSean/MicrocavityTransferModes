import numpy as np
from typing import Callable, List, Tuple
from scipy.optimize import curve_fit
from src.peak_relation import LabeledPeakCollection, LabeledPeakCluster


# (cavity_radius: float, wavelength: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
def get_radius_fit_function(cluster_array: List[LabeledPeakCluster], wavelength: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Theoretical self-consistency function based on transmission mode (q and m+n)
    :param cavity_radius: Cavity curvature (symmetric in 2 dimensions)
    :param wavelength: Resonating wavelength [nm]
    :return: Predicted resonance length depending on mode (q and m+n)
    """
    # d = np.asarray([cluster.get_avg_x for cluster in cluster_array])
    q = np.asarray([cluster.get_longitudinal_mode_id for cluster in cluster_array])
    m = np.asarray([cluster.get_transverse_mode_id for cluster in cluster_array])

    def map_function(d_array: np.ndarray, cavity_radius: float, length_offset: float) -> np.ndarray:
        """Return Theoretical self-consistency"""
        arcsin_input_array = np.sqrt(d_array / cavity_radius)
        result = (wavelength / 2) * (q + (m + 1) / np.pi * np.arcsin(arcsin_input_array)) - (d_array - length_offset)
        return result
    return map_function


def get_radius_estimate(cluster_array: List[LabeledPeakCluster], wavelength: float) -> Tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray([cluster.get_avg_x for cluster in cluster_array])
    fit_func = get_radius_fit_function(cluster_array=cluster_array, wavelength=wavelength)
    initial_radius = 10e4
    params, param_dict = curve_fit(fit_func, xdata=x_array, ydata=np.zeros(len(x_array)), p0=[initial_radius, 0.1], maxfev=100000, bounds=(0., +np.inf))
    return params, np.sqrt(np.diag(param_dict))  # [0], param_dict[0][0]  # mean, std


if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt
    from src.plot_functions import prepare_measurement_plot
    from src.peak_identifier import identify_peaks
    from src.peak_relation import get_converted_measurement_data
    from src.peak_normalization import NormalizedPeakCollection, NormalizedPeakCluster

    wavelength = 633  # nm
    # Collect peaks
    ax, measurement_class = prepare_measurement_plot('transrefl_hene_1s_10V_PMT4_rate1300000.0itteration1')
    measurement_class = get_converted_measurement_data(meas_class=measurement_class)
    labeled_collection = LabeledPeakCollection(identify_peaks(meas_data=measurement_class))
    # normal_collection = NormalizedPeakCollection(labeled_collection)
    cluster_array = labeled_collection.get_clusters
    [radius_mean, offset], [radius_std, offset_std] = get_radius_estimate(cluster_array=cluster_array, wavelength=wavelength)

    print(f'Cavity radius: R = {round(radius_mean, 1)} ' + r'+/-' + f' {round(radius_std, 1)} [nm]')
    print(f'Cavity offset length: {offset} [nm]')

    def plot_func(length_input: np.ndarray, trans_mode: int) -> np.ndarray:
        return (trans_mode / np.pi) * np.arcsin(np.sqrt((length_input + offset) / radius_mean))

    # Adds q clusters
    # NormalizedPeakCluster(data=cluster, anchor_data=labeled_collection)
    cluster_array.extend([cluster for cluster in labeled_collection.get_q_clusters])
    for i in [-1, 0, 1, 2, 3, 4, 5]:
        sub_cluster_array = [cluster for cluster in cluster_array if cluster.get_transverse_mode_id == i]
        if len(sub_cluster_array) == 0:
            continue

        d = np.asarray([cluster.get_avg_x for cluster in sub_cluster_array])

        y_array = plot_func(length_input=d, trans_mode=i)  # y_func(x_array, radius_mean)
        ax.plot(d, y_array, 'o', label=f'm+n={i}')

        x_space = np.linspace(-offset, d[-1], 100)
        y_space = plot_func(length_input=x_space, trans_mode=i)
        ax.plot(x_space, y_space, ('k:' if i == -1 else 'k-'))

    # Vertical indicator lines
    ax.axvline(x=-offset, ls='--', color='darkorange')
    ax.axvline(x=0, ls='--', color='darkorange')

    ax.set_title(f'Fitted cavity radius: R={round(radius_mean, 1)} ' + r'$\pm$' + f' {round(radius_std, 1)} [nm]')
    ax.set_ylabel(f'Transverse mode splitting' + r' [$\Delta L / (\lambda / 2)$]')
    ax.set_xlabel(f'Mirror position - {round(offset, 1)} (Offset)' + r' [nm]')
    ax.legend()
    plt.show()
