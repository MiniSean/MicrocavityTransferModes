import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, List, Tuple, Optional
from tqdm import tqdm  # For displaying for-loop process to console
from src.peak_relation import LabeledPeakCollection, LabeledPeakCluster, LabeledPeak


class ModeID:
    def __init__(self, long_id: int, trans_id: int, peak_id: Optional[int] = None):
        self.long_id = long_id
        self.trans_id = trans_id
        self.peak_id = peak_id

    @property
    def is_cluster(self) -> bool:
        return self.peak_id is None


class SplittingFocus:
    def __init__(self, data: List[Tuple[ModeID, ModeID]]):
        self._list = data

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


def plot_mode_splitting(collection_iterator: Iterable[LabeledPeakCollection], focus: SplittingFocus):
    result_dict = {}  # Dict[Tuple[Focus-q: int, target peak id: int], List[Peak distances: float]]
    for i, collection in tqdm(enumerate(collection_iterator), desc=f'Pre-Processing'):
        for mode_0, mode_1 in focus:
            # Find modes
            try:
                cluster_0 = collection.get_labeled_clusters(long_mode=mode_0.long_id, trans_mode=mode_0.trans_id)[0]
                cluster_1 = collection.get_labeled_clusters(long_mode=mode_1.long_id, trans_mode=mode_1.trans_id)[0]
            except IndexError:
                # Modes not found
                continue
            for j, peak in enumerate(cluster_1):
                distance = cluster_0.get_avg_x - peak.get_x
                key = (mode_0.long_id, j)
                if key not in result_dict:
                    result_dict[key] = [distance]
                else:
                    result_dict[key].append(distance)
        break
    # Plot
    _, _ax = plt.subplots()

    # Fetch data per peak id
    for peak_id in range(20):
        xs = []
        ys = []
        yerrs = []
        for mode_0, mode_1 in focus:
            key = (mode_0.long_id, peak_id)
            if key not in result_dict:
                break
            xs.append(key[0])
            ys.append(np.mean(result_dict[key]))
            yerrs.append(np.std(result_dict[key]))
        if len(xs) == 0:
            break
        _ax.errorbar(x=xs, y=ys, yerr=yerrs, fmt='.', label=f'ID:{peak_id}')

    _ax.set_ylabel(f'Distance [nm]',)
    _ax.set_xlabel(f'#Fundamental Mode')
    _ax.grid(True)
    _ax.legend()
    return _ax