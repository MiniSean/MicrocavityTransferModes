import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks, peak_prominences
from src.import_data import SyncMeasData
# cutoff = mean + c * std
PEAK_PROMINENCE = 0.3  # [0, 2]: Minimum peak-prominence value required to be detected


class PeakData(float):
    """
    Points to a specific location in the data where a significant peak in the data is identified.
    """

    def __new__(cls, data: SyncMeasData, index: int, *args, **kwargs):
        value = data.y_data[index]
        return float.__new__(cls, value)

    def __init__(self, data: SyncMeasData, index: int):
        self._data = data
        self._raw_index = index
        self._index_pointer = self._data.slicer[0] + index
        super(float).__init__()

    @property
    def relevant(self) -> bool:
        """Returns True if pointer is within the slice range of the data container."""
        return self._data.inside_global_slice_range(self._index_pointer)

    @property
    def get_x(self) -> float:
        """Returns the (converted) x value pointer within the data."""
        return self._data.x_boundless_data[self._index_pointer]

    @property
    def get_y(self) -> float:
        """Returns the y value pointer within the data."""
        return self._data.y_boundless_data[self._index_pointer]

    @property
    def get_relative_index(self) -> int:
        """Returns the relative index of the pointer location within the slice array."""
        return self._index_pointer - self._data.slicer[0]


class PeakCollection:
    """
    Array collection of PeakData class.
    """

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
    peak_prominence = identify_peak_prominence(meas_data)[0]  # Average peak prominence
    cutoff_prominence = np.mean(peak_prominence) + PEAK_PROMINENCE * np.std(peak_prominence)
    peak_indices, properties = find_peaks(x=meas_data.y_data, prominence=cutoff_prominence, distance=3)  # Arbitrary distance value
    peak_collection = PeakCollection([PeakData(data=meas_data, index=i) for i in peak_indices])
    return peak_collection


# Only for video analysis
def identify_peak_dirty(meas_data: SyncMeasData, cutoff: float = 0.4) -> PeakCollection:
    peak_prominence = identify_peak_prominence(meas_data)[0]  # Average peak prominence
    cutoff_prominence = np.mean(peak_prominence) + cutoff * np.std(peak_prominence)
    peak_indices, properties = find_peaks(x=meas_data.y_data, prominence=cutoff_prominence, distance=1)  # Arbitrary distance value
    peak_collection = PeakCollection([PeakData(data=meas_data, index=i) for i in peak_indices])
    return peak_collection


def identify_peak_prominence(meas_data: SyncMeasData) -> Tuple[np.ndarray, np.ndarray]:
    all_peaks_above_noise = find_peaks(x=meas_data.y_data)[0]
    return peak_prominences(x=meas_data.y_data, peaks=all_peaks_above_noise)
