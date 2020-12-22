import numpy as np
from typing import Optional, List
from src.import_data import FileToMeasData
from src.peak_identifier import identify_peaks
from src.peak_relation import LabeledPeakCluster, flatten_clusters, get_converted_measurement_data
from src.peak_normalization import NormalizedPeakCollection


class MeasurementAnalysis:

    def __init__(self, meas_file: str, samp_file: str, scan_file: Optional[str]):
        self._meas_file = meas_file
        self._samp_file = samp_file
        self._scan_file = scan_file
        # Construct synchronized measurement object
        self._meas_data = get_converted_measurement_data(FileToMeasData(meas_file=meas_file, samp_file=samp_file))
        self._peak_collection = NormalizedPeakCollection(identify_peaks(meas_data=self._meas_data))

    def __str__(self):
        result = f'--Structural analysis of Transmission measurement---'
        result += f'\n\n[File Info]:\nMeasurement: {self._meas_file}\nSampling: {self._samp_file}\nReference: {self._scan_file}'
        result += f'\n\n[Mode Identification Info]:\n{self.get_mode_info}'
        result += f'\n\n[Peak Identification Info]:\n{self.get_peak_info}'
        result += f'\n\n[Physical Info]:\n{self.get_physical_info}'
        return result

    @property
    def get_all_mode_clusters(self) -> List[LabeledPeakCluster]:
        return self._peak_collection.get_labeled_clusters(long_mode=None, trans_mode=None)

    @property
    def get_only_fundamental_clusters(self) -> List[LabeledPeakCluster]:
        return list(filter(lambda cluster: cluster.get_transverse_mode_id == 0, self.get_all_mode_clusters))

    @property
    def get_only_transverse_clusters(self) -> List[LabeledPeakCluster]:
        return list(filter(lambda cluster: cluster.get_transverse_mode_id != 0, self.get_all_mode_clusters))

    @property
    def get_max_transverse_mode(self) -> int:
        return sorted(self.get_all_mode_clusters, key=lambda cluster: cluster.get_transverse_mode_id)[-1].get_transverse_mode_id

    @property
    def get_mode_info(self) -> str:
        _fundamental_clusters = self._peak_collection.get_labeled_clusters(long_mode=None, trans_mode=0)
        result = f'Fundamental modes detected: {len(_fundamental_clusters)}'
        result += f'\nNumber of successful (m + n = -1) modes estimated: {len(list(self._peak_collection.q_dict.keys()))}'
        result += f'\nHighest order Transverse mode detected: m + n = {self.get_max_transverse_mode}'
        _transverse_only_cluster = self.get_only_transverse_clusters
        result += f'\nAverage Transverse modes per Fundamental mode: {round(len(_transverse_only_cluster) / len(_fundamental_clusters), 2)}'
        return result

    @property
    def get_peak_info(self) -> str:
        result = f'Average peaks detected per mode (m + n):'
        for _trans_mode in range(1, self.get_max_transverse_mode+1):
            _trans_cluster = self._peak_collection.get_labeled_clusters(long_mode=None, trans_mode=_trans_mode)
            _average_peaks_per_trans_mode = len(flatten_clusters(data=_trans_cluster)) / len(_trans_cluster)
            result += f'\n(m + n = {_trans_mode}) Average peaks: {round(_average_peaks_per_trans_mode, 2)}'
        return result

    @property
    def get_physical_info(self):
        _fundamental_only_cluster = self._peak_collection.get_q_clusters  # self.get_only_fundamental_clusters
        _fundamental_peak_distances = [abs(_fundamental_only_cluster[i].get_avg_x - _fundamental_only_cluster[i + 1].get_avg_x) for i in range(len(_fundamental_only_cluster) - 1)]
        avg_peak_dist = np.mean(_fundamental_peak_distances)
        std_peak_dist = np.std(_fundamental_peak_distances)
        result = f'Average fundamental peak distance: \u03BB/2 = {round(avg_peak_dist, 2)} [nm] (\u03BB = {round(2 * avg_peak_dist, 2)} \u00B1 {round(2 * std_peak_dist, 2)} [nm])'
        result += f'\nPredicted first resonance mode: q = {round(_fundamental_only_cluster[0].get_avg_x / avg_peak_dist)}'
        return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # File names
    file_meas = 'transrefl_hene_0_3s_10V_PMT4_rate1300000.0itteration0_pol000'
    file_samp = 'samples_0_3s_10V_rate1300000.0'
    # Analysis object
    analysis_obj = MeasurementAnalysis(meas_file=file_meas, samp_file=file_samp, scan_file=None)
    print(analysis_obj)
    plt.show()
