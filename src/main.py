#*************************************************************#
#    Analytic tool for optical resonance transmission data    #
#*************************************************************#
#
# Required data (transmission analysis)
# - transmission array (shape: m x 1)
# - sample (Voltage) array (shape: m x 1)
# - (Optional) reference transmission array (shape: m x 1)
#
# Required data (far-field analysis)
# - video file compatible with cv2 library (.mp4, .avi, etc.)
#
from src.import_data import FileToMeasData, DATA_DIR
from typing import Union, Iterator
from src.plot_functions import plot_peak_identification, plot_peak_relation, plot_peak_normalization_spectrum, plot_peak_normalization_overlap, plot_radius_estimate
from src.cavity_radius_analysis import get_radius_estimate
from src.peak_relation import get_converted_measurement_data
from src.peak_identifier import identify_peaks
from src.peak_relation import LabeledPeakCollection
from src.peak_normalization import NormalizedPeakCollection


def single_source_analysis(meas_file: str, samp_file: str, filepath: Union[str, None] = DATA_DIR):
    # Create measurement container instance
    measurement_container = FileToMeasData(meas_file=meas_file, samp_file=samp_file, filepath=filepath)

    # Apply voltage to length conversion
    measurement_container = get_converted_measurement_data(meas_class=measurement_container)

    # Peak Identification
    peak_collection = identify_peaks(meas_data=measurement_container)
    plot_peak_identification(collection=peak_collection, meas_class=measurement_container)  # Plot

    # Peak clustering and mode labeling
    labeled_collection = LabeledPeakCollection(transmission_peak_collection=peak_collection)
    plot_peak_relation(collection=labeled_collection, meas_class=measurement_container)  # Plot

    # Normalized based on free-spectral-ranges (FSR)
    normalized_collection = NormalizedPeakCollection(transmission_peak_collection=peak_collection)
    plot_peak_normalization_spectrum(collection=normalized_collection)  # Plot
    plot_peak_normalization_overlap(collection=normalized_collection)  # Plot

    # Estimate radius
    [radius_mean, offset], [radius_std, offset_std] = get_radius_estimate(cluster_array=labeled_collection.get_clusters, wavelength=633)
    plot_radius_estimate(collection=labeled_collection, radius_mean=radius_mean, offset=offset, radius_std=radius_std)  # Plot


def get_labeled_collection(meas_files: Iterator[str], samp_file: str, filepath: Union[str, None] = DATA_DIR) -> Iterator[LabeledPeakCollection]:
    for meas_file in meas_files:
        try:
            measurement_container = FileToMeasData(meas_file=meas_file, samp_file=samp_file, filepath=filepath)
            measurement_container = get_converted_measurement_data(meas_class=measurement_container)
            peak_collection = identify_peaks(meas_data=measurement_container)
            yield LabeledPeakCollection(transmission_peak_collection=peak_collection)
        except FileNotFoundError:
            continue


def multi_source_analysis(meas_files: Iterator[str], samp_file: str, filepath: Union[str, None] = DATA_DIR):
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Data file reference (Import data)
    file_path = 'data/Trans/20210104'  # Directory path from project root (Optional)
    file_meas = 'transrefl_hene_1s_10V_PMT4_rate1300000.0itteration0'
    file_samp = 'samples_1s_10V_rate1300000.0'

    single_source_analysis(meas_file=file_meas, samp_file=file_samp, filepath=file_path)

    plt.show()
