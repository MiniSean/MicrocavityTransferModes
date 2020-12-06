import numpy as np
from typing import List, Callable
import matplotlib.pyplot as plt


def quarter_phase_depth(resonance_wavelength: float, refraction_index: float):
    return resonance_wavelength / (4 * refraction_index)


def planar_transfer_matrix(depth: float, n: float) -> Callable[[float], np.ndarray]:
    """Returns lambda function depending on the wavelength"""
    return lambda x: np.array([[np.cos(2*np.pi*n*depth / x), -(1j/n)*np.sin(2*np.pi*n*depth / x)],
                               [-1j*n*np.sin(2*np.pi*n*depth / x), np.cos(2*np.pi*n*depth / x)]], dtype=complex)


def dielectric_layered_matrix(layers: int, n_1: float, n_2: float, resonance_wavelength: float) -> Callable[[float], np.ndarray]:
    m_01 = planar_transfer_matrix(quarter_phase_depth(resonance_wavelength, n_1), n_1)
    m_12 = planar_transfer_matrix(quarter_phase_depth(resonance_wavelength, n_2), n_2)

    def layered_matrix(wavelength: float) -> np.ndarray:
        result = np.identity(2)
        for i in range(layers):
            result = np.matmul(m_01(wavelength), result)
            result = np.matmul(m_12(wavelength), result)
        return result
    return lambda x: layered_matrix(x)


def layered_matrix(matrix_1: Callable[[float], np.ndarray], matrix_2: Callable[[float], np.ndarray], N: int) -> Callable[[float], np.ndarray]:
    def layered_matrix(wavelength: float) -> np.ndarray:
        result = np.identity(2)
        for i in range(N):
            result = np.matmul(matrix_1(wavelength), result)
            result = np.matmul(matrix_2(wavelength), result)
        return result
    return lambda x: layered_matrix(x)


def transfer_matrix_to_reflection(system_matrix: np.ndarray) -> float:
    # a = system_matrix[1][1] + system_matrix[0][1]
    # b = 1j*(system_matrix[1][1] - system_matrix[0][0])
    # c = system_matrix[1][1] - system_matrix[0][1]
    # d = 1j*(system_matrix[1][1] + system_matrix[0][0])
    # result = (a + b) / (c + d)
    e = system_matrix[0][0] + system_matrix[0][1]
    f = system_matrix[1][0] + system_matrix[1][1]
    result = (e - f) / (e + f)
    return float(result * np.conj(result))


def plot_wavelength_vs_reflection():
    resonance_wavelength = 640
    x = np.linspace(-500, 500, 100)
    x = resonance_wavelength + x  # x-axis scaling
    # Settings
    layers = 2
    n_1 = 1.46
    n_2 = 2.09
    system_matrix_function = dielectric_layered_matrix(layers, n_1, n_2, resonance_wavelength)
    y = [transfer_matrix_to_reflection(system_matrix_function(wavelength)) for wavelength in x]

    plt.plot(x, y)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    from src.import_data import import_npy
    from src.sample_conversion import fit_voltage_to_distance

    # Reference files
    file_samp = 'samples_1s_10V_rate1300000.0'
    filename_base = 'transrefl_tisaph_1s_10V_PMT4_rate1300000.0'

    data_array = import_npy(filename_base)[0]
    sample = import_npy(file_samp)

    # fig, ax = plt.subplots()
    # ax.plot(sample, data_array)
    # plt.show()

    result = fit_voltage_to_distance(voltage_array=sample, reference_transmission_array=data_array)
    print(result)