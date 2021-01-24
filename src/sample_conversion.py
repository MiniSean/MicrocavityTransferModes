import numpy as np
from scipy.optimize import curve_fit
import logging
from typing import Callable, Any, Iterator, List, Union
import matplotlib.pyplot as plt
from src.peak_relation import LabeledPeakCluster, SAMPLE_WAVELENGTH
FONT_SIZE = 20


# Laser-optics transformations
def length_to_transmission(wavelength: float, reflectance: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Calculates the round-trip phase change and resonator quality factor to construct the transmission intensity
    depending on the cavity length
    :param wavelength: Laser wavelength in vacuum [nm]
    :param reflectance: Cavity mirrors reflectance value
    :return: Callable function that maps cavity length [nm] to transmission value (0, 1]
    """
    # TODO: Might include penetration depth in the future)
    def map_function(d: np.ndarray) -> np.ndarray:
        phase_change = phase_addition(length=d, angle=0, n_refractive=1, wavelength=wavelength)
        q_factor = quality_factor(reflectance=reflectance)
        return 1. / (1 + q_factor * np.sin(0.5 * phase_change)**2)
    return map_function


def phase_addition(length: np.ndarray, angle: float, n_refractive: float, wavelength: float) -> np.ndarray:
    """
    Accumulating phase for single round-trip reflection inside cavity medium
    :param length: Medium length [nm]
    :param angle: incidence angle [radian]
    :param n_refractive: Cavity medium refractive index
    :param wavelength: Laser wavelength in vacuum [nm]
    :return: Accumulated phase change of light in medium [radians]
    """
    return (4 * np.pi / wavelength) * n_refractive * length * np.cos(angle)


def quality_factor(reflectance: float) -> float:
    """Dimensionless parameter for describing the under-damped classification of a resonator"""
    return (4. * reflectance) / (1 - reflectance)**2


# Piezoelectric transformations
def voltage_to_length(a: float, b: float, c: float, initial_length: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Third-order polynomial describing the piezo-electric displacement depending on voltage.
    With an initial cavity length the corrected cavity length is calculated.
    :param a: Third-order weight [nm / V^3]
    :param b: Second-order weight [nm / V^2]
    :param c: First-order weight [nm / V]

    Linear with exponential 'start-up' term.
    :param a: Exponential weight [1 / V]
    :param b: Voltage correction constant [V]
    :param c: First-order weight [nm / V]

    :param initial_length: Initial cavity length [nm]
    :return: Total cavity length [nm]
    """
    def map_function(v: np.ndarray) -> np.ndarray:
        # return initial_length - (a * v**3 + b * v**2 + c * v)
        return initial_length - (v * c) * (1 - np.exp(-v * a))
    return map_function


# (cavity_radius: float, wavelength: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
def length_mode_consistency(wavelength: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Theoretical self-consistency function based on transmission mode (q and m+n)
    :param cavity_radius: Cavity curvature (symmetric in 2 dimensions)
    :param wavelength: Resonating wavelength [nm]
    :return: Predicted resonance length depending on mode (q and m+n)
    """
    # (d: np.ndarray, q_mode: np.ndarray, m_mode: np.ndarray)
    def map_function(q_mode: np.ndarray) -> np.ndarray:
        """Return Theoretical self-consistency"""
        # arcsin_input = np.sqrt(np.abs(d / cavity_radius))
        # + (m_mode + 1) / np.pi * np.arcsin(arcsin_input)
        return (wavelength / 2) * q_mode
    return map_function


def fit_piezo_response(cluster_collection: List[LabeledPeakCluster], sample_wavelength: float, verbose: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """
    Attempts to fit a piezo-voltage to cavity-length response function based on the theoretical relations between the detected modes.
    :param cluster_collection: Collection with labled modes (q and m+n labels)
    :param sample_wavelength: Expected resonating light wavelength
    :return: Callable function that maps non-lineair voltage sample [V] to linear cavity length [nm]
    """
    # Initial condition
    # Polynomial parameters
    # a = -0.5
    # b = 13
    # c = 200
    # Exponential parameters
    a = 0.15
    b = 7.2
    c = 310

    # R = 2.7e4  # radius of curvature
    L0 = 3900
    p0 = [a, b, c, L0]

    # Mode samples
    q_offset = 3
    x_array = np.asarray([mode.get_avg_x for mode in cluster_collection])
    q = np.asarray([mode.get_longitudinal_mode_id + q_offset for mode in cluster_collection])
    # m = np.asarray([mode.get_transverse_mode_id for mode in cluster_collection])

    # Fit function
    def fit_func(voltage_array: np.ndarray, _a: float, _b: float, _c: float, _L0: float):
        """Return function to fit to 0: length = cavity_formula(length) + L1 (self consistency)"""
        length = voltage_to_length(a=_a, b=_b, c=_c, initial_length=_L0)(voltage_array)
        theory_length = length_mode_consistency(wavelength=sample_wavelength)(q)
        return length - theory_length

    # Fitting
    # logging.warning(f'Start voltage to nm conversion fitting (using q* = q - {q_offset})')
    popt, pcov = curve_fit(fit_func, xdata=x_array, ydata=np.zeros(len(x_array)), p0=p0, maxfev=100000)
    a, b, c, L0 = popt

    # # Temp
    # print(popt)
    # a, b, c, L0, L1 = popt
    # print(f'Estimation parameters:')
    print(f'Cavity initial length: {L0} [nm]')

    # Plot
    voltage_map = lambda v: voltage_to_length(a=a, b=b, c=c, initial_length=L0)(v)
    if verbose:
        cavity_map = lambda d: length_mode_consistency(wavelength=sample_wavelength)(q)
        plot_response_mapping(cluster_collection=cluster_collection, q_offset=q_offset, fit_function=lambda x: fit_func(x, a, b, c, L0), length_map=voltage_map, cavity_map=cavity_map)

    print(f'Cavity offset length (lowest q*={int(q_offset)}): {np.min(voltage_map(x_array))} [nm]')
    # return Piezo behaviour function
    # logging.warning(f'Finished conversion fitting')
    return voltage_map


# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)


def plot_response_mapping(cluster_collection: List[LabeledPeakCluster], q_offset: int, fit_function: Callable[[np.ndarray], np.ndarray], length_map: Callable[[np.ndarray], np.ndarray], cavity_map: Callable[[np.ndarray], np.ndarray]):
    # Mode samples
    x_array = np.asarray([mode.get_avg_x for mode in cluster_collection])
    x_std_array = np.asarray([mode.get_std_x for mode in cluster_collection])
    # Plot settings
    dot_fmt = '.'
    dot_color = 'b'
    line_width = 0.5
    cap_width = 1
    # Plot deviation from theory
    # fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    # Effective difference with uncertainty
    avg_length_array = fit_function(x_array)
    max_length_array = fit_function(np.add(x_array, x_std_array))
    min_length_array = fit_function(np.add(x_array, -x_std_array))
    yerr_length_array = np.array([np.add(avg_length_array, - max_length_array), np.add(min_length_array, -avg_length_array)])
    # Plot effective difference
    ax2.errorbar(x=x_array, y=avg_length_array, yerr=yerr_length_array, fmt=dot_fmt, linewidth=line_width, capsize=cap_width)
    ax2.set_title(f'Effective difference (q = q* + {q_offset})', fontsize=FONT_SIZE)
    ax2.set_title(f'(c)', fontsize=FONT_SIZE)
    ax2.set_xlabel('Voltage [V]', fontsize=FONT_SIZE)
    ax2.set_ylabel('$L_{cav} - L_{qmn}$ [nm]', fontsize=FONT_SIZE)
    ax2.grid(True)

    # Plot cavity length
    # Fitted length with uncertainty
    avg_fitted_array = length_map(x_array)
    max_fitted_array = length_map(np.add(x_array, x_std_array))
    min_fitted_array = length_map(np.add(x_array, -x_std_array))
    yerr_fitted_array = np.array([np.add(avg_fitted_array, -max_fitted_array), np.add(min_fitted_array, -avg_fitted_array)])
    # Theoretical cavity length with uncertainty
    avg_cavity_array = cavity_map(avg_fitted_array)
    max_cavity_array = cavity_map(max_fitted_array)
    min_cavity_array = cavity_map(min_fitted_array)
    yerr_cavity_array = np.array([np.add(avg_cavity_array, -max_cavity_array), np.add(min_cavity_array, -avg_cavity_array)])
    # ax1.plot(x_array, avg_cavity_array, '.')
    ax1.errorbar(x=x_array, y=avg_cavity_array, yerr=yerr_cavity_array, fmt=dot_fmt, color=dot_color, linewidth=line_width, capsize=cap_width)
    ax1.set_title(f'Theory prediction based on fitted cavity length', fontsize=FONT_SIZE)
    ax1.set_title(f'(b)', fontsize=FONT_SIZE)
    ax1.set_xlabel('Voltage [V]', fontsize=FONT_SIZE)
    ax1.set_ylabel('$L_{qmn}$ [nm]', fontsize=FONT_SIZE)
    ax1.grid(True)
    # fig, ax2 = plt.subplots()
    # ax0.plot(x_array, avg_fitted_array, '.')
    ax0.errorbar(x=x_array, y=avg_fitted_array, yerr=yerr_fitted_array, fmt='-', color=dot_color, linewidth=line_width, capsize=cap_width)
    ax0.set_title(f'Fitted cavity length', fontsize=FONT_SIZE)
    ax0.set_title(f'(a)', fontsize=FONT_SIZE)
    ax0.set_xlabel('Voltage [V]', fontsize=FONT_SIZE)
    ax0.set_ylabel('$L_{cav}$ [nm]', fontsize=FONT_SIZE)
    ax0.grid(True)


def fit_calibration(voltage_array: np.ndarray, reference_transmission_array: np.ndarray, response_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    wavelength_tisaph = 794  # [nm]

    # Initial condition
    p0 = [0, 0.0, 1]

    def calibration_curve(voltage: np.ndarray, _A: float, _R: float, L0: float):  # _B: float
        return _A + length_to_transmission(wavelength=wavelength_tisaph, reflectance=_R)(L0 + response_func(voltage))

    # Fitting
    popt, pcov = curve_fit(calibration_curve, xdata=voltage_array, ydata=reference_transmission_array, p0=p0, maxfev=100000)

    # Plot deviation from theory
    fig, ax0 = plt.subplots()
    ax0.plot(voltage_array, reference_transmission_array, color='orange')
    ax0.plot(voltage_array, calibration_curve(voltage_array, popt[0], popt[1], popt[2]), color='b')
    ax0.set_xlabel('Voltage [V]')
    ax0.set_ylabel('Transmission + offset [a.u.]')
    return popt


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.import_data import import_npy, FileToMeasData
    from src.peak_identifier import identify_peaks
    from src.peak_relation import LabeledPeakCollection
    file_samp = 'samples_1s_10V_rate1300000.0'
    file_meas = 'transrefl_hene_1s_10V_PMT4_rate1300000.0itteration0'  # 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration5'  #
    filename_base = 'transrefl_tisaph_1s_10V_PMT4_rate1300000.0'

    # Define font
    plt.rcParams.update({'font.size': FONT_SIZE})

    data_class = FileToMeasData(meas_file=file_meas, samp_file=file_samp)
    collection_class = LabeledPeakCollection(identify_peaks(meas_data=data_class))

    cluster_collection = collection_class.get_q_clusters  # collection_class.get_clusters
    piezo_response = fit_piezo_response(cluster_collection=cluster_collection, sample_wavelength=SAMPLE_WAVELENGTH)
    # piezo_response = fit_collection()
    # fit_variables = fit_calibration(voltage_array=data_class.samp_array, reference_transmission_array=import_npy(filename_base)[0], response_func=piezo_response)
    # print(f'TiSaph transmission: T = {1 - fit_variables[1]} (R = {fit_variables[1]})')
    # print(f'Cavity length delta between HeNe and TiSaph measurement: {fit_variables[2]} [nm]')

    for i in range(5):
        _filename = 'transrefl_hene_1s_10V_PMT4_rate1300000.0itteration{}'.format(i)
        data_class = FileToMeasData(meas_file=_filename, samp_file=file_samp)
        identified_peaks = identify_peaks(meas_data=data_class)
        collection_class = LabeledPeakCollection(identified_peaks)

        cluster_collection = collection_class.get_q_clusters  # collection_class.get_clusters
        piezo_response = fit_piezo_response(cluster_collection=cluster_collection, sample_wavelength=SAMPLE_WAVELENGTH, verbose=True)
    # Obtain mean and root-mean-square
    y_values = [value for line in ax2.lines for value in line.get_ydata()]
    y_mean = np.mean(y_values)
    y_rms = np.sqrt(np.sum((y_values - y_mean)**2) / len(y_values))
    ax2.axhline(y=y_mean, ls='--', color='darkorange')
    ax2.axhline(y=y_mean+y_rms, ls='--', color='orange', label=r'$\mu + \sigma$' + f': {round(y_mean+y_rms, 2)} [nm]')
    ax2.axhline(y=y_mean-y_rms, ls='--', color='orange', label=r'$\mu - \sigma$' + f': {round(y_mean-y_rms, 2)} [nm]')

    # plt.tight_layout(pad=.01)
    plt.legend()
    plt.show()
