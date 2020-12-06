import numpy as np
from scipy.optimize import curve_fit
# import logging
from typing import Callable, Any, Iterator
import matplotlib.pyplot as plt


# Laser-optics transformations
def transmission_to_length():
    pass


def length_to_transmission(angle: float, n_refractive: float, wavelength: float, reflectance: float) -> Callable[[float], float]:
    """
    Calculates the round-trip phase change and resonator quality factor to construct the transmission intensity
    depending on the cavity length
    :param angle: incidence angle [radian]
    :param n_refractive: Cavity medium refractive index
    :param wavelength: Laser wavelength in vacuum [nm]
    :param reflectance: Cavity mirrors reflectance value
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
def voltage_to_length(max_length: float, slope: float, offset: float) -> Callable[[float], float]:
    """
    Estimates the piezo voltage to length by a third order polynomial
    :param max_length: Sigmoid cap
    :param slope: Sigmoid slope
    :param offset: Temporal shift (5V at 10V max)
    :return: Sigmoid displacement function
    """
    def sigmoid(v: float) -> float:
        return max_length / (1 + np.exp(-slope * (v - offset)))

    def map_linear(v: float) -> float:
        return (v / (2 * offset) - (0.003 * v**2)) * max_length
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

    return map_linear


def length_to_voltage():
    pass


def fit_voltage_to_distance(voltage_array: np.ndarray, reference_transmission_array: np.ndarray):
    """4th Edition Optics Eugene Hecht. page 419"""

    # def fit_function(voltage: np.ndarray, b1, b2, b3, A, B, C) -> Callable[[np.ndarray, Any], np.ndarray]:
    #     wl = 794  # Reference laser frequency
    #     return A + B / (1 + C * np.cos(np.pi * 2 / (wl / 2) * (b1 * voltage + b2 + b1 * b3 * voltage ** 2)))

    wavelength = 794  # nm
    half_max_voltage = 5  # [V]

    def fit_function(length: np.ndarray, angle: float, n_refractive: float, reflectance: float) -> Iterator[float]:
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
        # displacement_calculator = voltage_to_length(max_length=max_length * wavelength, slope=slope, offset=half_max_voltage + offset)
        transmission_calculator = length_to_transmission(angle=angle, n_refractive=n_refractive, wavelength=wavelength, reflectance=reflectance)
        # length = np.fromiter(map(displacement_calculator, voltage), dtype=np.float)
        return np.fromiter(map(transmission_calculator, length), dtype=np.float)  # map(displacement_calculator, voltage)

    initial_condition = [0., 1., .7]  #, 7., 1., 0.]  # Angle, n, R, max_len, slope  # [300, 600, -0.03, 0.4, 0.07, -0.7]
    length_array = np.fromiter(map(voltage_to_length(max_length=7 * wavelength, slope=.5, offset=5.), voltage_array), dtype=np.float)  # Temp
    popt, pvoc = curve_fit(f=fit_function, xdata=length_array, ydata=reference_transmission_array, p0=initial_condition)
    # Temp
    # yout = fit_function(voltage_array, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
    yout = fit_function(voltage_array, popt[0], popt[1], popt[2])
    plt.plot(voltage_array, reference_transmission_array)
    plt.plot(voltage_array, yout)
    plt.show()
    return popt


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Test length to transmission function
    w = 794
    max_len = 5*w
    map_func = length_to_transmission(angle=0., n_refractive=1, wavelength=w, reflectance=.9)
    map_piezo = voltage_to_length(max_length=max_len, slope=.5, offset=5.)

    x = np.linspace(0, max_len, 10000)
    p = np.linspace(0, 10, 10000)
    y = np.fromiter(map(map_func, map(map_piezo, p)), dtype=np.float)
    q = np.fromiter(map(map_piezo, p), dtype=np.float) / max_len
    plt.plot(p, y)
    plt.plot(p, q)
    plt.show()
