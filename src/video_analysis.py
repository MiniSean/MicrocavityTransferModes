import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm  # For displaying for-loop process to console
from src.import_data import import_cv2, ICAP_DIR, SyncMeasData, import_npy
from src.peak_identifier import identify_peak_dirty


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


def export_video_intensity(filename: str, update_capture_images: bool):
    folder_name = os.path.join(ICAP_DIR, f'FrameFolder_{filename}')
    capture = import_cv2(filename=filename)
    ret, frame = capture.read()
    max_frame_nr = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Number of frames recorded: {max_frame_nr} ({round((max_frame_nr / 15) / 60, 2)} min)')

    # Ready data storage
    storage = np.zeros(max_frame_nr)
    storage_filename = f'NumpyData_{filename}'

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

    data_array = np.nan_to_num(storage)
    data_array /= np.max(np.abs(data_array), axis=0)
    sub_threshold_indices = data_array < 0.0  # TODO: Hardcoded (normalized) intensity threshold
    data_array[sub_threshold_indices] = 0

    # Create data object
    data_class = SyncMeasData(data_array=data_array, samp_array=np.arange(len(storage)))
    peak_collection = identify_peak_dirty(meas_data=data_class, cutoff=0.5)  # TODO: Hardcoded peak identification prominence
    print(f'Number of high intensity peaks detected: {len(peak_collection)}')
    for i, peak_data in enumerate(peak_collection):
        if i > 1000:  # safety break
            break
        if peak_data.relevant:
            plt.plot(peak_data.get_x, peak_data.get_y, 'x', color='r', alpha=1)

    plt.plot(data_array)
    plt.yscale('log')
    plt.show()

    if update_capture_images:
        # Continue processing images
        for peak in tqdm(peak_collection, desc=f'Retrieve prominent frames'):
            i = int(peak.get_x)
            # # Analyse frame
            capture.set(cv2.CAP_PROP_POS_FRAMES, i+1)
            _, frame_i = capture.read()
            plt.imshow(frame_i / frame_i.max())
            frame_name = f'frame_{str(i)}'
            export_plt(plot_obj=plt, filename=frame_name, filepath=folder_name)
            plt.close()
    # Closing
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # filename = '07-12_800sec_mode_scanning_15FPS'  # 'test_01'
    filename = 'pol040_exp005_gain800'
    export_video_intensity(filename=filename, update_capture_images=False)
