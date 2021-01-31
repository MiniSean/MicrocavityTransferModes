import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm  # For displaying for-loop process to console
from src.import_data import import_cv2, ICAP_DIR, SyncMeasData, import_npy
from src.peak_identifier import identify_peak_dirty, identify_peaks, PeakCollection
from src.peak_relation import LabeledPeakCollection


def export_plt(plot_obj: plt, filename: str, filepath: str) -> str:
    """
    Exports single frames from pyplot object.
    :param plot_obj: pyplot object to store frame from.
    :param filename: Desired file name of the *.png.
    :param filepath: Desired file path to store the frame.
    :return: Full storage path of the new *.png file (with extension)
    """
    sub_folder = filepath  # os.path.join(ICAP_DIR, filepath)
    if not os.path.exists(sub_folder):  # Creates folder if folder does not exist
        os.makedirs(sub_folder)
    full_path = os.path.join(sub_folder, filename + '.png')
    plot_obj.savefig(full_path)
    return full_path


def export_npy(npy_object: np.ndarray, filename: str, filepath: str) -> str:
    """
    Exports single frames from pyplot object.
    :param npy_object: numpy object to store.
    :param filename: Desired file name of the *.npy.
    :param filepath: Desired file path to store the numpy data.
    :return: Full storage path of the new *.npy file (without extension)
    """
    sub_folder = filepath  # os.path.join(ICAP_DIR, filepath)
    if not os.path.exists(sub_folder):  # Creates folder if folder does not exist
        os.makedirs(sub_folder)
    full_path = os.path.join(sub_folder, filename)
    np.save(exclude_file_extension(full_path=full_path), npy_object)
    return full_path


def exclude_file_extension(full_path: str) -> str:
    return os.path.splitext(full_path)[0]


def post_processing(frame: np.ndarray, base_frame: np.ndarray) -> np.ndarray:
    """Applies non-destructive filtering"""
    # mean_cutoff = np.mean(base_frame)  # Mean cutoff
    # frame = ndimage.gaussian_filter(frame, 3)  # High frequency noise filter
    #
    # sub_threshold_indices = frame < mean_cutoff  # Apply base cutoff
    # frame[sub_threshold_indices] = 0
    frame = np.clip(frame - base_frame, 0, 225)
    frame = frame / frame.max()  # Normalization
    return frame


def export_video_intensity(filename: str, update_capture_images: bool, build_video: bool, full_sweep: bool):
    split_directory = filename.split('\\')
    actual_filename = split_directory[-1]
    actual_filedir = '\\'.join(split_directory[0:-1])
    folder_name = os.path.join(os.path.join(ICAP_DIR, actual_filedir), f'FrameFolder_{actual_filename}')
    capture = import_cv2(filename=filename)
    ret, frame = capture.read()
    max_frame_nr = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(f'Number of frames recorded: {max_frame_nr} ({round((max_frame_nr / fps) / 60, 2)} min)')

    # Ready data storage
    storage = np.zeros(max_frame_nr)
    storage_filename = f'NumpyData_{actual_filename}'

    # Store/Retrieve video intensity information
    try:
        storage = import_npy(filename=storage_filename, filepath=folder_name)
    except FileNotFoundError:
        # Iterate through frames
        for i in tqdm(range(max_frame_nr), desc='Collect frame intensities for data storage'):
            _, frame_i = capture.read(i)
            storage[i] = np.nan_to_num(np.sum(frame_i))
        storage = np.asarray(storage)
        export_npy(npy_object=storage, filename=storage_filename, filepath=folder_name)

    # Background filter
    filter_start_frame = 20
    filter_range = 100
    full_capt_start = 2450  # Temp
    full_capt_end = 2530  # Temp

    data_array = np.nan_to_num(storage)
    data_array /= np.max(np.abs(data_array), axis=0)
    sub_threshold_indices = data_array < 0.0  # TODO: Hardcoded (normalized) intensity threshold
    data_array[sub_threshold_indices] = 0

    # Create data object
    data_class = SyncMeasData(data_array=data_array, samp_array=np.arange(len(storage)))
    peak_collection = identify_peak_dirty(meas_data=data_class, cutoff=0.0)  # TODO: Hardcoded peak identification prominence
    peak_collection = PeakCollection([peak for peak in peak_collection if peak.get_y > 0.34])
    # peak_collection = LabeledPeakCollection(transmission_peak_collection=peak_collection)
    print(f'Number of high intensity peaks detected: {len(peak_collection)}')
    plot_peaks(plt, peak_collection)

    plt.plot(data_array)
    plt.axvline(x=filter_start_frame, color='r')
    plt.axvline(x=filter_start_frame + filter_range, color='r')
    plt.axvline(x=full_capt_start, color='darkorange')  # Temp
    plt.axvline(x=full_capt_end, color='darkorange')  # Temp
    plt.yscale('log')
    plt.show()

    # Base reference frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
    _, frame_0 = capture.read()
    frame_0 = np.ndarray(shape=frame_0.shape)

    # Construct highlight capture images
    if update_capture_images:
        # Get average background for subtraction
        for i in range(filter_start_frame, filter_start_frame + filter_range):
            # # Analyse frame
            capture.set(cv2.CAP_PROP_POS_FRAMES, i+1)
            _, frame_i = capture.read()
            frame_0 += frame_i
        frame_0 = frame_0 / filter_range  # Normalize

        if full_sweep:
            # Get all frames in range
            for i in tqdm(range(full_capt_start, full_capt_end), desc=f'Retrieve full range of frames'):
                if i % 2 == 0:
                    continue
                # # Analyse frame
                capture.set(cv2.CAP_PROP_POS_FRAMES, i+1)
                _, frame_i = capture.read()
                frame_i = post_processing(frame=frame_i, base_frame=frame_0)

                fig, axis = plt.subplots()
                axis.get_xaxis().set_visible(False)  # Hide tick labels
                axis.get_yaxis().set_visible(False)  # Hide tick labels
                inset_axis = add_inset_spectrum(figure=fig)
                track_axis = add_track_spectrum(figure=fig)
                axis.imshow(frame_i)
                # Create navigation spectrum inset
                set_inset_spectrum(axis=inset_axis, data=data_array, current_index=i, peak_collection=peak_collection)
                set_track_spectrum(axis=track_axis, data=data_array, current_index=i)

                # plt.show()  # Temp
                frame_name = f'frame_{str(i)}'  #_bright
                plot_filepath = export_plt(plot_obj=plt, filename=frame_name, filepath=folder_name)
                plt.close()
        else:
            # Continue processing images
            for peak in tqdm(peak_collection, desc=f'Retrieve prominent frames'):
                j = int(peak.get_x)
                j_margin = 0
                for i in range(j - j_margin, j + j_margin + 1):
                    # Analyse frame
                    capture.set(cv2.CAP_PROP_POS_FRAMES, i+1)
                    _, frame_i = capture.read()
                    frame_i = post_processing(frame=frame_i, base_frame=frame_0)

                    fig, axis = plt.subplots()
                    axis.get_xaxis().set_visible(False)  # Hide tick labels
                    axis.get_yaxis().set_visible(False)  # Hide tick labels
                    inset_axis = add_inset_spectrum(figure=fig)
                    track_axis = add_track_spectrum(figure=fig)
                    axis.imshow(frame_i)
                    # Create navigation spectrum inset
                    set_inset_spectrum(axis=inset_axis, data=data_array, current_index=i, peak_collection=peak_collection)
                    set_track_spectrum(axis=track_axis, data=data_array, current_index=i)

                    # plt.show()  # Temp
                    frame_name = f'frame_{str(i)}'  #_bright
                    plot_filepath = export_plt(plot_obj=plt, filename=frame_name, filepath=folder_name)
                    plt.close()

    # Create video
    if build_video:
        # output_name = f'Highlight_test.avi'  # {actual_filename}
        # cmd = f'ffmpeg -framerate {fps} -pattern_type glob -i {os.path.join(folder_name, "*.png")} {output_name}'
        # process = subprocess.Popen(cmd)
        size = (frame_0.shape[1], frame_0.shape[0])
        output_filepath = os.path.join(folder_name, f'Highlight_{actual_filename}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_filepath, fourcc, fps, size)
        # Buffer image file path
        plot_image_buffer = [os.path.join(folder_name, f'frame_{str(int(peak.get_x))}.png') for peak in peak_collection]
        for path in tqdm(reversed(plot_image_buffer), desc=f'Construct highlight video'):
            frame = cv2.imread(filename=path)
            video.write(frame)  # Append video
        video.release()

    # Closing
    capture.release()
    cv2.destroyAllWindows()


def capture_frame(capture: cv2.VideoCapture, data_array: np.ndarray, index: int, frame_0: np.ndarray, folder_name: str, peak_collection: PeakCollection):
    # # Analyse frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, index+1)
    _, frame_i = capture.read()
    frame_i = post_processing(frame=frame_i, base_frame=frame_0)

    fig, axis = plt.subplots()
    axis.get_xaxis().set_visible(False)  # Hide tick labels
    axis.get_yaxis().set_visible(False)  # Hide tick labels
    inset_axis = add_inset_spectrum(figure=fig)
    track_axis = add_track_spectrum(figure=fig)
    axis.imshow(frame_i)
    # Create navigation spectrum inset
    set_inset_spectrum(axis=inset_axis, data=data_array, current_index=index, peak_collection=peak_collection)
    set_track_spectrum(axis=track_axis, data=data_array, current_index=index)

    # plt.show()  # Temp
    frame_name = f'frame_{str(index)}'  #_bright
    plot_filepath = export_plt(plot_obj=plt, filename=frame_name, filepath=folder_name)
    plt.close()


def construct_video(filename: str):
    split_directory = filename.split('\\')
    actual_filename = split_directory[-1]
    actual_filedir = '\\'.join(split_directory[0:-1])
    folder_name = os.path.join(os.path.join(ICAP_DIR, actual_filedir), f'FrameFolder_{actual_filename}')

    # Fetch images
    for img_name in glob.glob(os.path.join(folder_name, '*.png')):
        print(img_name)


def plot_peaks(axis: plt.axis, collection: PeakCollection) -> plt.axis:
    for i, peak_data in enumerate(collection):
        if i > 1000:  # safety break
            break
        if peak_data.relevant:
            axis.plot(peak_data.get_x, peak_data.get_y, 'x', color='r', alpha=1)
    return axis


def add_inset_spectrum(figure: plt.Figure) -> plt.axis:
    rect_transform = [.7, .7, .25, .25]  # (0,0 is bottom left; 1,1 is top right)
    _axis = figure.add_axes(rect_transform)
    _axis.get_xaxis().set_visible(False)  # Hide tick labels
    _axis.get_yaxis().set_visible(False)  # Hide tick labels
    _axis.set_yscale('log')  # Set log scale
    return _axis


def add_track_spectrum(figure: plt.Figure) -> plt.axis:
    rect_transform = [0.125, 0.05, .775, .15]  # (0,0 is bottom left; 1,1 is top right)
    _axis = figure.add_axes(rect_transform)
    _axis.get_xaxis().set_visible(False)  # Hide tick labels
    _axis.get_yaxis().set_visible(False)  # Hide tick labels
    _axis.set_yscale('log')  # Set log scale
    _axis.set_facecolor('k')  # Face color
    return _axis


def set_inset_spectrum(axis: plt.axis, data: np.ndarray, current_index: int, peak_collection: LabeledPeakCollection) -> plt.axis:
    axis.plot(data)  # Show intensity spectrum
    axis = plot_peaks(axis=axis, collection=peak_collection)  # Show peaks
    y_bot, y_top = axis.get_ylim()
    text_height = y_bot + 0.6 * (y_top - y_bot)  # Data coordinates
    _margin = 50
    x_lim = [max(0, current_index - _margin), min(len(data) - 1, current_index + _margin)]
    # Plot clusters
    # for cluster in peak_collection.get_clusters:
    #     bound_left, bound_right = cluster.get_value_slice
    #     if bound_right > x_lim[0] or bound_left < x_lim[1]:
    #         axis.axvspan(bound_left, bound_right, alpha=0.5, color='green')
    #     if x_lim[0] < cluster.get_avg_x < x_lim[1]:
    #         axis.text(x=cluster.get_avg_x, y=text_height, s=r'$\tilde{m}$'+f'={cluster.get_transverse_mode_id}')
    axis.axvline(x=current_index, color='r')
    axis.set_xlim(x_lim)
    return axis


def set_track_spectrum(axis: plt.axis, data: np.ndarray, current_index: int) -> plt.axis:
    axis.plot(data)  # Show intensity spectrum
    axis.axvline(x=current_index, color='r')
    # axis.set_xlim([max(0, current_index - _margin), min(len(data) - 1, current_index + _margin)])
    # axis.set_ylim([max(0.001, np.min(data)), np.max(data)])
    return axis


if __name__ == '__main__':

    # filename = 'test_01'
    # filename = '07-12_800sec_mode_scanning_15FPS'  # 'test_01'
    # filename = 'pol040_exp005_gain800'
    # filename = '21-12_10micromcav\\pol000_exp005_gain800_04'
    # filename = '21-12_10micromcav\\pol000_fps0375_gain1000_01HIGHQUALITY'
    # filename = '21-12_10micromcav\\pol000_fps0750_gain1000_01'
    # filename = '27-01_12micromcav\\TM03\\V683_689_G800_15FPS'
    # filename = '27-01_12micromcav\\TM03\\V683_689_G800_15FPS_P045'
    filename = '27-01_12micromcav\\TM03_LowQ\\V703_711_G800_15FPS_P067'
    export_video_intensity(filename=filename, update_capture_images=True, build_video=False, full_sweep=True)
