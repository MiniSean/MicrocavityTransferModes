import numpy as np
from scipy.optimize import curve_fit
# import logging
from typing import Callable, Any, Iterator, List
import matplotlib.pyplot as plt


# Laser-optics transformations
def transmission_to_length():
    pass


def length_to_transmission(wavelength: float, reflectance: float, angle: float = 0, n_refractive: float = 1) -> Callable[[float], float]:
    """
    Calculates the round-trip phase change and resonator quality factor to construct the transmission intensity
    depending on the cavity length
    :param wavelength: Laser wavelength in vacuum [nm]
    :param reflectance: Cavity mirrors reflectance value
    :param angle: incidence angle [radian]
    :param n_refractive: Cavity medium refractive index
    :return: Callable function that maps cavity length [nm] to transmission value (0, 1]
    """
    # length: float, :param length: Cavity length [nm] (TODO: Might include penetration depth in the future)
    def map_function(d: float) -> float:
        phase_change = phase_addition(length=d, angle=angle, n_refractive=n_refractive, wavelength=wavelength)
        q_factor = quality_factor(reflectance=reflectance)
        return 1. / (1 + q_factor * np.sin(0.5 * phase_change)**2)
    return map_function


def phase_addition(length: float, angle: float, n_refractive: float, wavelength: float) -> float:
    """Accumulating phase for single round-trip reflection inside cavity"""
    return (2. * np.pi / wavelength) * 2 * n_refractive * length * np.cos(angle)


def quality_factor(reflectance: float) -> float:
    """Dimensionless parameter for describing the under-damped classification of a resonator"""
    return (4. * reflectance) / (1 - reflectance)**2


# Piezoelectric transformations
def voltage_to_length(a: float, b: float = 0, c: float = 0) -> Callable[[float], float]:

    def map_exponential(v: float) -> float:
        return a * v + b*(np.exp(-c * v) - 1)

    # def map_linear(v: float) -> float:
    #     return a * (v**2) + b * v + c

    # """
    # Estimates the piezo voltage to length by a third order polynomial
    # (max_length: float, slope: float, offset: float) -> Callable[[float], float]:
    # :param max_length: Sigmoid cap
    # :param slope: Sigmoid slope
    # :param offset: Temporal shift (5V at 10V max)
    # :return: Sigmoid displacement function
    # """
    # def sigmoid(v: float) -> float:
    #     return max_length / (1 + np.exp(-slope * (v - offset)))

    # """
    # Estimates the piezo voltage to length by a third order polynomial
    # (coef_con: float, coef_lin: float, coef_qua: float, coef_cub: float) -> Callable[[float], float]:
    # :param coef_con: Constant order coefficient [nm]
    # :param coef_lin: Linear order coefficient
    # :param coef_qua: Quadratic order coefficient
    # :param coef_cub: Cubic order coefficient
    # :return: Displacement [nm]
    # """
    # def map_function(v: float) -> float:
    #     return coef_cub * v ** 3 + coef_qua * v ** 2 + coef_lin * v + coef_con

    return map_exponential


def length_to_voltage():
    pass


def fit_voltage_to_distance(voltage_array: np.ndarray, reference_transmission_array: np.ndarray):
    """4th Edition Optics Eugene Hecht. page 419"""

    # def fit_function(voltage: np.ndarray, b1, b2, b3, A, B, C) -> Callable[[np.ndarray, Any], np.ndarray]:
    #     wl = 794  # Reference laser frequency
    #     return A + B / (1 + C * np.cos(np.pi * 2 / (wl / 2) * (b1 * voltage + b2 + b1 * b3 * voltage ** 2)))

    wavelength = 794  # nm

    def fit_length(voltage: np.ndarray) -> np.ndarray:
        lin_coef = 1000
        L0 = max(voltage * lin_coef)
        # cavity_length = lin_coef * voltage + (np.exp(-voltage * exp_coef) - 1)  # Exponential
        cavity_length = L0 - voltage * lin_coef  # Linear
        return cavity_length

    def fit_function(voltage: np.ndarray, reflectance: float, meas_offset: float) -> np.ndarray:  # L0: float, lin_coef: float, , exp_coef: float
        """
        Fit function that combines the piezo voltage to displacement function and the Fabry-Perot resonance length to transmission function.
        (voltage: np.ndarray, angle: float, n_refractive: float, reflectance: float, max_length: float, slope: float, offset: float) -> Iterator[float]:
        :param voltage: Input voltage
        :param angle: incidence angle [radian]
        :param n_refractive: Cavity medium refractive index
        :param reflectance: Cavity mirrors reflectance value
        :param max_length: Sigmoid cap
        :param slope: Sigmoid slope
        :param offset: Temporal shift
        :return: Transmission Output
        """
        phase_change = (4. * np.pi / wavelength) * fit_length(voltage)
        q_factor = (4. * reflectance) / (1 - reflectance)**2
        transmission = 1. / (1 + q_factor * np.sin(0.5 * phase_change)**2)
        return transmission + meas_offset
        #
        # displacement_calculator = voltage_to_length(a=a)
        # transmission_calculator = length_to_transmission(wavelength=wavelength, reflectance=reflectance)
        # length = L0 - np.fromiter(map(displacement_calculator, voltage), dtype=np.float)
        # return np.fromiter(map(transmission_calculator, length), dtype=np.float)

    initial_condition = [.5, 0.]  #  7.25 * wavelength, 0.5 * wavelength,
    # bounds = ([0.0, 0, 0, -10., 0.], [1.0, 15 * wavelength, 1.5 * wavelength, 10., 1.])  # (lower, upper) bounds
    popt, pvoc = curve_fit(f=fit_function, xdata=voltage_array, ydata=reference_transmission_array, p0=initial_condition)  # , bounds=bounds)

    # Display length approximation
    fig, ax = plt.subplots()
    ax.plot(voltage_array, fit_length(voltage_array))
    ax.set_title(f'Cavity length over voltage')
    ax.set_xlabel('Sampling Voltage [V]')
    ax.set_ylabel('Cavity Length [nm]')

    fig, ax2 = plt.subplots()
    ax2.plot(voltage_array, reference_transmission_array)
    ax2.plot(voltage_array, fit_function(voltage_array, popt[0], popt[1]))
    ax2.set_title(f'Transmission fit')
    ax2.set_xlabel('Cavity Length [nm]')
    ax2.set_ylabel('Transmission [a.u.]')
    return popt


def fit_collection():
    wavelength_hene = 633  # [nm]

    # Initial condition
    b1 = 250
    b4 = 300
    b5 = 0.01
    R = 27e3  # radius of curvature
    L0 = 3400
    L1 = 600
    p0 = [b1, b4, b5, R, L0, L1]

    # Peak samples
    x_array = np.asarray([peak.get_x for peak in collection_class])
    q = np.asarray([peak.get_longitudinal_mode_id for peak in collection_class])
    m = np.asarray([peak.get_transverse_mode_id for peak in collection_class])

    def length_func(samples: np.ndarray, _b1: float, _b4: float, _b5: float, _L0: float):
        return _L0 - (_b1 * samples + _b4 * (np.exp(-samples * _b5) - 1))  # Exponential piezo displacement

    def cavity_func(length: np.ndarray, _R: float, _wavelength: float):
        return wavelength_hene / 2 * (q + (m + 1) / np.pi * np.arcsin(np.sqrt(np.abs(length / _R))))

    def fit_func(samples: np.ndarray, _b1: float, _b4: float, _b5: float, _R: float, _L0: float, _L1: float):
        length = length_func(samples, _b1, _b4, _b5, _L0)  # Exponential piezo displacement
        cavity_formula = cavity_func(length, _R, wavelength_hene)  # Cavity length calculation
        return length - cavity_formula - L1

    # Fitting
    popt, pcov = curve_fit(fit_func, xdata=x_array, ydata=np.zeros(len(x_array)), p0=p0, maxfev=100000)

    # Plot deviation from theory
    b1, b4, b5, R, L0, L1 = popt
    fig, ax0 = plt.subplots()
    ax0.plot(x_array, fit_func(x_array, b1, b4, b5, R, L0, L1), '.', color='b')
    ax0_twinx = ax0.twinx()
    ax0_twinx.plot(x_array, fit_func(x_array, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5]), '.', color='orange')  # reference
    ax0.set_xlabel('Voltage [V]')
    ax0.set_ylabel('Deviation from theory [nm]', color='b')
    ax0_twinx.set_ylabel('Pre-fit total length [nm]', color='orange')

    # Plot cavity length
    fig, ax1 = plt.subplots()
    length_array = length_func(x_array, b1, b4, b5, L0) - L1
    cavity_array = cavity_func(length_array, R, wavelength_hene)
    ax1.plot(x_array, cavity_array, '.')
    fig, ax2 = plt.subplots()
    ax2.plot(x_array, length_array, '.')
    ax2.set_xlabel('Voltage [V]')
    ax2.set_ylabel('$L_{cav}$ [nm]')


def fit_calibration(voltage_array: np.ndarray, reference_transmission_array: np.ndarray) -> np.ndarray:
    wavelength_tisaph = 794  # [nm]

    # Initial condition
    p0 = [270, 0, -0.00, 0.1, 0.02, -0.7]

    def calibration_curve(voltage: np.ndarray, _b1: float, _b2: float, _b3: float, _A: float, _B: float, _C: float):
        return _A + _B / (1 + _C * np.cos((4 * np.pi / wavelength_tisaph) * (_b1 * voltage + _b2 + _b1 * _b3 * voltage ** 2)))

    def calibration_curve_exp(voltage: np.ndarray, _b1: float, _b2: float, _b3: float, _A: float, _B: float, _C: float):
        # Q_factor = quality_factor(reflectance=_C)
        return _A + _B / (1 + _C * np.cos((4 * np.pi / wavelength_tisaph) * (_b1 * voltage + _b2 * (np.exp(-_b3 * voltage) - 1))))

    # Fitting
    popt, pcov = curve_fit(calibration_curve_exp, xdata=voltage_array, ydata=reference_transmission_array, p0=p0, maxfev=100000)

    # Plot deviation from theory
    fig, ax0 = plt.subplots()
    ax0.plot(voltage_array, calibration_curve_exp(voltage_array, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]))
    ax0.plot(voltage_array, reference_transmission_array)
    ax0.set_xlabel('Voltage [V]')
    ax0.set_xlabel('Transmission [a.u.]')
    return popt


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.import_data import import_npy, SyncMeasData
    from src.peak_identifier import identify_peaks
    from src.peak_relation import LabeledPeakCollection
    file_samp = 'samples_1s_10V_rate1300000.0'
    file_meas = 'transrefl_hene_1s_10V_PMT5_rate1300000.0_pol000'
    filename_base = 'transrefl_tisaph_1s_10V_PMT4_rate1300000.0'

    data_class = SyncMeasData(meas_file=file_meas, samp_file=file_samp, scan_file=None)
    collection_class = LabeledPeakCollection(identify_peaks(meas_data=data_class))

    fit_variables = fit_calibration(voltage_array=data_class.samp_array, reference_transmission_array=import_npy(filename_base)[0])
    print(fit_variables)
    # fit_collection()
    plt.show()

    # data_array = import_npy(filename_base)[0]
    # sample = import_npy(file_samp)
    # # Test length to transmission function
    # w = 794
    # max_len = 5*w
    # map_func = length_to_transmission(angle=0., n_refractive=1, wavelength=w, reflectance=.5)
    # map_piezo = voltage_to_length(a=100., b=50, c=300)
    #
    # x = np.linspace(0, max_len, 10000)
    # p = np.linspace(0, 10, 10000)
    # distance = 7.25*w - np.fromiter(map(map_piezo, p), dtype=np.float)
    # y = np.fromiter(map(map_func, distance), dtype=np.float)
    # q = distance / max_len
    # plt.plot(p, y)
    # plt.plot(p, q)
    # plt.plot(sample, data_array)
    # plt.show()
