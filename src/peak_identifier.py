import numpy as np
from scipy.signal import find_peaks, peak_prominences
from typing import List, Tuple, Dict


class PeakData:
    def __init__(self):
        pass


def identify_peaks(meas_data: np.ndarray) -> Tuple[np.ndarray, Dict]:  # -> List[PeakData]:
    mean_prominence = np.mean(identify_peak_prominence(meas_data)[0])  # Average peak prominence
    return find_peaks(x=meas_data, height=identify_noise_ceiling(meas_data), prominence=mean_prominence, distance=10)  # Arbitrary distance value


def identify_noise_ceiling(meas_data: np.ndarray) -> float:
    mean = np.mean(meas_data)
    std = np.std(meas_data)
    return mean + .1 * std  #TODO: still hardcoded. Need to find a satisfying way of representing noise ceiling


def identify_peak_prominence(meas_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    all_peaks_above_noise = find_peaks(x=meas_data, height=identify_noise_ceiling(meas_data))[0]
    return peak_prominences(x=meas_data, peaks=all_peaks_above_noise)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.plot_npy import prepare_measurement_plot, plot_class
    # Construct measurement class
    ax, measurement_class = prepare_measurement_plot()
    # Optional, define slice
    slice = (1050000, 1150000)
    measurement_class.slicer = slice  # Zooms in on relevant data part

    # Collect noise ground level
    noise_ceiling = identify_noise_ceiling(measurement_class.y_data)
    # Collect peaks
    peak_indices = identify_peaks(measurement_class.y_data)

    # Apply axis draw/modification
    ax = plot_class(axis=ax, measurement_class=measurement_class)

    # Draw noise h-lines
    ax.axhline(y=noise_ceiling, color='r')
    # Draw peak v-lines
    print(len(peak_indices[0]))
    for i, index in enumerate(peak_indices[0]):
        if i > 1000:  # safety break
            break
        ax.axvline(x=measurement_class.x_data[index], color='r', alpha=0.5)  # ymax=measurement_class.y_data[index],

    # Show
    plt.show()
