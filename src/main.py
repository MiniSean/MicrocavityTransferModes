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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.import_data import FileToMeasData
    from src.plot_functions import plot_peak_identification, plot_peak_relation

    # Data file reference (Import data)
    file_path = 'data/Trans/20210104'  # Directory path from project root (Optional)
    file_meas = 'transrefl_hene_1s_10V_PMT4_rate1300000.0itteration0'
    file_samp = 'samples_1s_10V_rate1300000.0'

    # Create measurement container instance
    measurement_container = FileToMeasData(meas_file=file_meas, samp_file=file_samp, filepath=file_path)

    # Apply voltage to length conversion
    from src.peak_relation import get_converted_measurement_data
    measurement_container = get_converted_measurement_data(meas_class=measurement_container)

    # Peak Identification
    from src.peak_identifier import identify_peaks
    peak_collection = identify_peaks(meas_data=measurement_container)
    plot_peak_identification(collection=peak_collection, meas_class=measurement_container)  # Plot

    # Peak clustering and mode labeling
    from src.peak_relation import LabeledPeakCollection
    labeled_collection = LabeledPeakCollection(transmission_peak_collection=peak_collection)
    plot_peak_relation(collection=labeled_collection, meas_class=measurement_container)  # Plot

    # Normalized based on free-spectral-ranges (FSR)
    from src.peak_normalization import NormalizedPeakCollection
    normalized_collection = NormalizedPeakCollection(transmission_peak_collection=peak_collection)

    plt.show()
