import numpy as np
from typing import Callable, List, Tuple
from scipy.optimize import curve_fit
from src.peak_relation import LabeledPeakCluster


# (cavity_radius: float, wavelength: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
def get_radius_fit_function(cluster_array: List[LabeledPeakCluster], wavelength: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Theoretical self-consistency function based on transmission mode (q and m+n).
    :param cluster_array: Array of mode clusters.
    :param wavelength: Resonating wavelength [nm].
    :return: Predicted resonance length depending on mode (q and m+n).
    """
    q = np.asarray([cluster.get_longitudinal_mode_id for cluster in cluster_array])
    m = np.asarray([cluster.get_transverse_mode_id for cluster in cluster_array])

    def map_function(d_array: np.ndarray, cavity_radius: float, length_offset: float) -> np.ndarray:
        """Return Theoretical self-consistency function."""
        arcsin_input_array = np.sqrt(d_array / cavity_radius)
        result = (wavelength / 2) * (q + (m + 1) / np.pi * np.arcsin(arcsin_input_array)) - (d_array - length_offset)
        return result
    return map_function


def get_radius_estimate(cluster_array: List[LabeledPeakCluster], wavelength: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a fit of the cavity radius based on the transmission mode cluster spreading."""
    x_array = np.asarray([cluster.get_avg_x for cluster in cluster_array])
    fit_func = get_radius_fit_function(cluster_array=cluster_array, wavelength=wavelength)
    initial_radius = 10e4
    params, param_dict = curve_fit(fit_func, xdata=x_array, ydata=np.zeros(len(x_array)), p0=[initial_radius, 0.1], maxfev=100000, bounds=(0., +np.inf))
    return params, np.sqrt(np.diag(param_dict))  # [0], param_dict[0][0]  # mean, std


def get_gouy_func(length_input: np.ndarray, length_offset: float, radius_mean: float, trans_mode: int) -> np.ndarray:
    """Returns Gouy approximation based on cavity length (offset), estimated radius and transverse mode identity."""
    return ((trans_mode + 1) / np.pi) * np.arcsin(np.sqrt((length_input + length_offset) / radius_mean))
