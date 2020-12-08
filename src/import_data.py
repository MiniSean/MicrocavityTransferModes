import os
import numpy as np
# import logging
from typing import Tuple, Optional, List
DATA_DIR = 'data/Trans/20201125'


# Synchronized measurement data
class SyncMeasData:
    """
    Includes:
     -transmitted signal detection
     -sample scaling
     -background high wavelength laser scan (outside cavity stopband) to accurately determine cavity length)
    """
    def __init__(self, meas_file: str, samp_file: str, scan_file: Optional[str]):
        self.data_array = import_npy(meas_file)[0]  # Contains both transmitted as reflected data
        self.samp_array = import_npy(samp_file)
        # Sort data and sample array
        self.samp_array, [self.data_array] = SyncMeasData.sort_arrays(leading=self.samp_array, follow=[self.data_array])
        self._slicer_bounds = (0, min(len(self.y_boundless_data), len(self.x_boundless_data)))  # For clamping
        self._slicer = self._slicer_bounds

    @staticmethod
    def sort_arrays(leading: np.ndarray, follow: [np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        follow_result = []
        for follow_array in follow:
            follow_result.append(np.asarray([x for _, x in sorted(zip(leading, follow_array), key=lambda pair: pair[0])]))
        return np.asarray(sorted(leading, key=lambda x: x)), follow_result

    # Getters
    def get_transmitted_data(self) -> np.ndarray:
        return self.data_array

    def get_sample_data(self) -> np.ndarray:
        return self.samp_array

    def get_sliced_transmitted_data(self) -> np.ndarray:
        return self.get_sliced_array(self.get_transmitted_data())

    def get_sliced_sample_data(self) -> np.ndarray:
        return self.get_sliced_array(self.get_sample_data())

    def get_slicer(self) -> Tuple[int, int]:
        return self._slicer

    # Setters
    def set_slicer(self, new_slicer: Tuple[int, int]):
        self._slicer = new_slicer
        if self._slicer[0] > self._slicer[1]:  # Swap data_slice order to maintain (low, high) bound structure
            self._slicer = (self._slicer[1], self._slicer[0])
        # Clamp slicer
        self._slicer = (max(self._slicer_bounds[0], self._slicer[0]), min(self._slicer_bounds[1], self._slicer[1]))

    # Properties
    y_boundless_data = property(get_transmitted_data)  # Boundless data only for peak reference
    x_boundless_data = property(get_sample_data)  # Boundless data only for peak reference
    # For all other references
    y_data = property(get_sliced_transmitted_data)
    x_data = property(get_sliced_sample_data)
    slicer = property(get_slicer, set_slicer)

    def get_sliced_array(self, array: np.ndarray):
        try:
            return slice_array(array=array, slice=self._slicer)
        except TypeError:
            return array

    def inside_global_slice_range(self, index: int) -> bool:
        return (self.slicer[0] <= index) and (index <= self.slicer[1])


def import_npy(filename: str) -> np.ndarray:
    filepath = os.path.join(DATA_DIR, filename + '.npy')
    return np.load(file=filepath, allow_pickle=False)


def slice_array(array: np.ndarray, slice: Tuple[int, int]) -> np.ndarray:
    if not isinstance(slice, Tuple) or len(slice) != 2:
        raise TypeError(f'Slice does not have correct type. Expected {type(Tuple)}, got {type(slice)}.')
    min_index = max((0, min(slice)))  # min(data_slice)
    max_index = min((len(array)-1, max(slice)))  # max(data_slice)
    return array[min_index: max_index]
