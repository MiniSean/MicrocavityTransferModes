import os
import numpy as np
# import logging
from typing import Tuple, Optional
DATA_DIR = 'data/Trans'


# Synchronized measurement data
class SyncMeasData:
    """
    Includes:
     -transmitted signal detection
     -sample scaling
     -background high wavelength laser scan (outside cavity stopband) to accurately determine cavity length)
    """
    def __init__(self, meas_file: str, samp_file: str, scan_file: Optional[str]):
        self.data_array = import_npy(meas_file)  # Contains both transmitted as reflected data
        self.samp_array = import_npy(samp_file)
        self._slicer = None

    # Getters
    def get_transmitted_data(self):
        return self.get_sliced_array(self.data_array[0])

    def get_sample_data(self):
        return self.get_sliced_array(self.samp_array)

    # Setters
    def set_slicer(self, new_slicer: Tuple[int, int]):
        self._slicer = new_slicer

    # Properties
    y_data = property(get_transmitted_data)
    x_data = property(get_sample_data)
    slicer = property(None, set_slicer)

    def get_sliced_array(self, array: np.ndarray):
        try:
            return slice_array(array=array, slice=self._slicer)
        except TypeError:
            return array


def import_npy(filename: str) -> np.ndarray:
    filepath = os.path.join(DATA_DIR, filename + '.npy')
    return np.load(file=filepath, allow_pickle=False)


def slice_array(array: np.ndarray, slice: Tuple[int, int]) -> np.ndarray:
    if not isinstance(slice, Tuple) or len(slice) != 2:
        raise TypeError(f'Slice does not have correct type. Expected {type(Tuple)}, got {type(slice)}.')
    min_index = max((0, min(slice)))  # min(slice)
    max_index = min((len(array)-1, max(slice)))  # max(slice)
    return array[min_index: max_index]
