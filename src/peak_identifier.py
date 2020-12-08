import numpy as np
from typing import List, Tuple, Dict, Union
from scipy.signal import find_peaks, peak_prominences
from src.import_data import SyncMeasData


class PeakData(float):

    def __new__(cls, data: SyncMeasData, index: int, *args, **kwargs):
        value = data.y_data[index]
        return float.__new__(cls, value)

    def __init__(self, data: SyncMeasData, index: int):
        # (self, loc: float, height: float)
        self._data = data
        self._raw_index = index
        self._index_pointer = self._data.slicer[0] + index
        super(float).__init__()

    @property
    def relevant(self) -> bool:
        return self._data.inside_global_slice_range(self._index_pointer)

    @property
    def get_x(self) -> float:
        return self._data.x_boundless_data[self._index_pointer]

    @property
    def get_y(self) -> float:
        return self._data.y_boundless_data[self._index_pointer]

    @property
    def get_relative_index(self) -> int:
        return self._index_pointer - self._data.slicer[0]

    def get_x_offset(self, offset_index: int):
        return self._data.x_boundless_data[self._index_pointer + offset_index]


class PeakCollection:
    def __init__(self, collection: List[PeakData]):
        self._list = collection
        self._global_offset = 0  # Temp

    def __iter__(self):
        return self._list.__iter__()

    def __len__(self):
        return self._list.__len__()

    def __getitem__(self, item):
        return self._list.__getitem__(item)

    def __setitem__(self, key, value):
        return self._list.__setitem__(key, value)

    def index(self, *args, **kwargs):
        return self._list.index(*args, **kwargs)


def identify_peaks(meas_data: SyncMeasData) -> PeakCollection:
    mean_prominence = np.mean(identify_peak_prominence(meas_data)[0])  # Average peak prominence
    peak_indices, properties = find_peaks(x=meas_data.y_data, height=identify_noise_ceiling(meas_data), prominence=mean_prominence, distance=10)  # Arbitrary distance value
    peak_collection = PeakCollection([PeakData(data=meas_data, index=i) for i in peak_indices])
    return peak_collection


def identify_noise_ceiling(meas_data: SyncMeasData) -> float:
    mean = np.mean(meas_data.y_boundless_data)
    std = np.std(meas_data.y_boundless_data)
    return .003  #mean + .24 * std  # TODO: still hardcoded. Need to find a satisfying way of representing noise ceiling


def identify_peak_prominence(meas_data: SyncMeasData) -> Tuple[np.ndarray, np.ndarray]:
    all_peaks_above_noise = find_peaks(x=meas_data.y_data, height=identify_noise_ceiling(meas_data))[0]
    return peak_prominences(x=meas_data.y_data, peaks=all_peaks_above_noise)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.plot_npy import prepare_measurement_plot, plot_class
    # Construct measurement class
    ax, measurement_class = prepare_measurement_plot('transrefl_hene_0_3s_10V_PMT5_rate1300000.0itteration0_pol000')
    # Optional, define data_slice
    # data_slice = (1050000, 1150000)
    # measurement_class.slicer = data_slice  # Zooms in on relevant data part

    # Collect noise ground level
    noise_ceiling = identify_noise_ceiling(measurement_class)
    # Collect peaks
    peak_collection = identify_peaks(measurement_class)

    # data_slice = (1090000, 1120000)
    # measurement_class.slicer = data_slice  # Zooms in on relevant data part
    # Apply axis draw/modification
    ax = plot_class(axis=ax, measurement_class=measurement_class)

    # Draw noise h-lines
    ax.axhline(y=noise_ceiling, color='r')
    # Draw peak v-lines
    print(len(peak_collection))
    for i, peak_data in enumerate(peak_collection):
        if i > 1000:  # safety break
            break
        if peak_data.relevant:
            ax.plot(peak_data.get_x, peak_data.get_y, 'o', color='r', alpha=1)
        # ax.axvline(x=peak_data.get_x, ymax=peak_data.get_y, color='r', alpha=1)

    # Show
    plt.show()
