from typing import Iterator, Tuple, Callable
import numpy as np
from src.peak_relation import SAMPLE_WAVELENGTH


def get_mode_groups(trans_mode: int) -> Iterator[Tuple[int, int]]:
    """Returns all (p, l) groups available in transverse mode N."""
    l_array = np.arange(0, trans_mode + 1, dtype=int)
    p_array = np.arange(0, (trans_mode + 1)/2, dtype=int)
    for p in p_array:
        for l in l_array:
            if trans_mode == 2*p + l:
                yield (p, l)


def get_predicted_cavity_length_function(cav_length: float, cav_radius: float) -> Callable[[int, Tuple[int, int]], Iterator[float]]:
    _gouy_phase = np.arcsin(np.sqrt(cav_length / cav_radius))

    def map_function(long_mode: int, mode_group: Tuple[int, int]) -> Iterator[float]:
        """Returns predicted cavity length based on q + (p, l) group"""
        _p, _l = mode_group
        _k = 1 / SAMPLE_WAVELENGTH
        # + _l
        for sign in [+1, -1]:
            _higher_order = (2 * _p ** 2 + 2 * _p * _l - _l ** 2 + 2 * _p + _l + sign * 4 * _l - 2) / (8 * np.pi * _k * cav_radius)
            yield (SAMPLE_WAVELENGTH / 2) * (long_mode + 1 + ((2 * _p + _l + 1) * _gouy_phase) / np.pi + _higher_order)

        # - _l
        if _l != 0:
            _l = -_l
            for sign in [+1, -1]:
                _higher_order = (2 * _p ** 2 + 2 * _p * _l - _l ** 2 + 2 * _p + _l + sign * 4 * _l - 2) / (8 * np.pi * _k * cav_radius)
                yield (SAMPLE_WAVELENGTH / 2) * (long_mode + 1 + ((2 * _p + _l + 1) * _gouy_phase) / np.pi + _higher_order)

    return map_function


if __name__ == '__main__':

    q_length = 3002   # nm
    radius = 12000  # nm
    length_map = get_predicted_cavity_length_function(cav_length=length, cav_radius=radius)
    for p, l in get_mode_groups(trans_mode=3):
        print(p, l)
        print(length_map(8, (p, l)))
