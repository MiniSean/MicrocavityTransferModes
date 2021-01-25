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
from src.import_data import FileToMeasData, DATA_DIR, import_npy
from typing import Union, Iterable
from src.plot_functions import plot_peak_identification, plot_peak_relation, plot_peak_normalization_spectrum, plot_peak_normalization_overlap, plot_radius_estimate, plot_allan_variance
from src.plot_iteration_mode_color import plot_pinned_focus_top, plot_pinned_focus_side
from src.cavity_radius_analysis import get_radius_estimate
from src.peak_relation import get_converted_measurement_data
from src.peak_identifier import identify_peaks
from src.peak_relation import LabeledPeakCollection
from src.peak_normalization import NormalizedPeakCollection
from src.structural_analysis import MeasurementAnalysis
from src.plot_mode_splitting import plot_mode_splitting, SplittingFocus, ModeID
from src.allan_variance_analysis import get_allan_variance
Q_OFFSET = 7


def single_source_analysis(meas_file: str, samp_file: str, filepath: Union[str, None] = DATA_DIR):
    # Create measurement container instance
    measurement_container = FileToMeasData(meas_file=meas_file, samp_file=samp_file, filepath=filepath)

    # Apply voltage to length conversion
    measurement_container = get_converted_measurement_data(meas_class=measurement_container, q_offset=Q_OFFSET)

    # Peak Identification
    peak_collection = identify_peaks(meas_data=measurement_container)
    # plot_peak_identification(collection=peak_collection, meas_class=measurement_container)  # Plot

    # Peak clustering and mode labeling
    labeled_collection = LabeledPeakCollection(transmission_peak_collection=peak_collection, q_offset=Q_OFFSET)
    plot_peak_relation(collection=labeled_collection, meas_class=measurement_container)  # Plot
    #
    # # Normalized based on free-spectral-ranges (FSR)
    # normalized_collection = NormalizedPeakCollection(transmission_peak_collection=peak_collection)
    # plot_peak_normalization_spectrum(collection=normalized_collection)  # Plot
    # plot_peak_normalization_overlap(collection=normalized_collection)  # Plot
    #
    # # Estimate radius
    # [radius_mean, offset], [radius_std, offset_std] = get_radius_estimate(cluster_array=labeled_collection.get_clusters, wavelength=633)
    # plot_radius_estimate(collection=labeled_collection, radius_mean=radius_mean, offset=offset, radius_std=radius_std)  # Plot
    #
    # # Analysis object
    # analysis_obj = MeasurementAnalysis(meas_file=file_meas, samp_file=file_samp, collection=normalized_collection)
    # print(analysis_obj)


def get_file_fetch_func(file_base_name: str, filepath: Union[str, None] = DATA_DIR):
    """Returns a function that accepts an interation index and tries to fetch the corresponding file."""

    def file_fetch_function(iteration: int) -> Union[str, FileNotFoundError]:
        _filename = file_base_name.format(iteration)
        try:
            import_npy(filename=_filename, filepath=filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f'File ({_filename}) does not exist in pre-defined directory')
        return _filename

    return file_fetch_function


def get_labeled_collection(meas_file_base: str, iter_count: Iterable[int], samp_file: str, filepath: Union[str, None] = DATA_DIR) -> Iterable[LabeledPeakCollection]:
    """Returns Iterable for LabeledPeakCluster data."""
    fetch_func = get_file_fetch_func(file_base_name=meas_file_base, filepath=filepath)
    for i in iter_count:
        try:
            meas_file = fetch_func(iteration=i)
            measurement_container = FileToMeasData(meas_file=meas_file, samp_file=samp_file, filepath=filepath)
            measurement_container = get_converted_measurement_data(meas_class=measurement_container, q_offset=Q_OFFSET)
            yield LabeledPeakCollection(transmission_peak_collection=identify_peaks(meas_data=measurement_container), q_offset=Q_OFFSET)
        except FileNotFoundError:
            continue


def multi_source_analysis(meas_file_base: str, iter_count: Iterable[int], samp_file: str, filepath: Union[str, None] = DATA_DIR):

    # Peak clustering and mode labeling
    iterable_collections = list(get_labeled_collection(meas_file_base=meas_file_base, iter_count=iter_count, samp_file=samp_file, filepath=filepath))

    # # Top-view measurement transmission comparison (with pin and focus selection)
    # pin/focus = ( longitudinal mode ID, transverse mode ID )
    # plot_pinned_focus_top(collection_iterator=iterable_collections, pin=(9, 0), focus=[(8, 8)])  # Plot
    #
    # # Side-view measurement transmission focus
    plot_pinned_focus_side(collection_iterator=iterable_collections, pin=(10, 8))

    # Mode splitting
    plot_mode_splitting(collection_iterator=iterable_collections, focus=global_focus)  # Plot

    # # Allan variance
    # scan_speed = 3500
    # _x, allan_variance_y = get_allan_variance(collection_iterator=iterable_collections, scan_velocity=scan_speed)
    # plot_allan_variance(xs=_x, ys=allan_variance_y)  # Plot


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Data file reference (Import data)
    file_path = 'data/Trans/20210120'  # Directory path from project root (Optional)
    file_meas = 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration0'
    file_samp = 'samples_1s_10V_rate1300000.0'

    # Single measurement analysis tools
    # single_source_analysis(meas_file=file_meas, samp_file=file_samp, filepath=file_path)

    # Data file reference (Import data)
    file_meas_base = 'transrefl_hene_1s_10V_PMT5_rate1300000.0itteration{}'

    # Multi measurement analysis tools
    global_focus = SplittingFocus(data=[(ModeID(9, 0), ModeID(8, 8, 0)), (ModeID(10, 0), ModeID(9, 8, 0)), (ModeID(11, 0), ModeID(10, 8, 0))])
    multi_source_analysis(meas_file_base=file_meas_base, iter_count=range(8), samp_file=file_samp, filepath=file_path)

    plt.show()
