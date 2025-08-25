import math
import numpy as np
import pandas as pd
import shutil
import os
import csv
import constants as const
from tqdm import tqdm
from Utils import (
    format_mars_frames,
    normalize_data,
    OfflineManager,
    format_single_frame_mode,
    relative_coordinates,
    format_batched_frames,
)
from Tracking import (
    TrackBuffer,
    BatchedData,
)
from wakepy import keep

# TODO: Make this constant as the environment setup 
# Dataset capturing environment setup offsets
KINECT_Z = 0.7
KINECT_X = 0.15
RELATIVE_ENABLED = True

# Retrieves frame no. with -ve head z value ( kinect data reporting multiple bodies)
def fix_kinect(data):
    d = {}
    for _, row in data.iterrows():
        if int(row[0]) not in d:
            d[int(row[0])] = []
        d[int(row[0])].append(row.tolist())
    ignore_frames = []
    head_idx = 3*3 +2
    for ts in d.keys():
        if len(d[ts]) > 1:
            for b in d[ts]:
                head_z = b[head_idx + 1]
                if head_z < 0.1:
                    ignore_frames.append(int(b[1]))
    return ignore_frames


# Pairs the kinect and mmwave frames together
# I/p: Experiment name (A1, A2...)
# O/p: [(mmwave_frame_no, kinect_frame_no),...]
def pair(experiment):
    kinect_input = os.path.join(
        f"{const.P_LOG_PATH}{const.P_KINECT_DIR}", f"{experiment}.csv"
    )
    mmwave_input = os.path.join(f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}", experiment)

    df2 = pd.read_csv(kinect_input, header=None)
    kinect_ignore = fix_kinect(df2)
    df2 = df2[~df2.iloc[:, 1].isin(kinect_ignore)]
    pairs = []
    filenames = os.listdir(mmwave_input)
    filenames_sorted = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    for filename in filenames_sorted:
        with open(os.path.join(mmwave_input, filename), "r") as file:
            df1 = pd.read_csv(file, header=None)
            unique_frames = df1.drop_duplicates(subset=0)

            for _, row1 in unique_frames.iterrows():
                timestamp1 = row1[6]
                closest_row = df2.iloc[(df2[0] - timestamp1).abs().argsort()[:1]] #Find the closest row in kinect wrt to mmwave timestamp
                timestamp2 = closest_row.iloc[0, 0]

                if abs(timestamp1 - timestamp2) < 20: #If timestamp difference between 20ms, then include the pair ids 
                    pairs.append((int(row1[0]), closest_row.iloc[0, 1]))
    return pairs


def filter_kinect_frames(pairs, invalid_frames, experiment):
    input_file = os.path.join(
        f"{const.P_LOG_PATH}{const.P_KINECT_DIR}", f"{experiment}.csv"
    )
    output_file = os.path.join(
        f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}", f"{experiment}.csv"
    )
    centroid_dict = {}
    invalid_kinect_frames = []
    for inv_frame in invalid_frames:
        for pair in pairs:
            if pair[0] == inv_frame:
                invalid_kinect_frames.append(pair[1])

    with open(input_file, "r", newline="") as infile, open(
        output_file, "w", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        valid_counter = 0
        for rows in reader:

            if len(rows) > 60:
                row = [item for i, item in enumerate(rows) if i not in [20,21,22, 32,33,34]]
            else:
                row = rows

            if any(int(row[1]) == f_pair[1] for f_pair in pairs) and not any(
                int(row[1]) == inv_frame1 for inv_frame1 in invalid_kinect_frames
            ):

                translated_row = translate_kinect(row)

                skeleton_spinebase = [ float(translated_row[2]), float(translated_row[4]), float(translated_row[3]) ]
                centroid_dict[row[1]] = skeleton_spinebase

                if RELATIVE_ENABLED:
                    translated_row = static_kinect(translated_row)


                writer.writerow(translated_row)
                valid_counter += 1
    return centroid_dict

def translate_kinect(row, kinect_x = None, kinect_z = None):
    # TODO: Is this angle rad for the tilt of the mmwave radar?
    ang_rad = np.radians(0)
    z, y = 0, 0
    for i in range(2, len(row) - 1):
        # For all x coords
        if i % 3 == 2:
            row[i] = str(float(row[i]) + (kinect_x if kinect_x is not None else KINECT_X))
            z = float(row[i + 1])
            y = float(row[i + 2])

        # For all z coords:
        elif i % 3 == 0:
            row[i] = str(
                y * np.sin(ang_rad) + float(row[i]) * np.cos(ang_rad) + (kinect_z if kinect_z is not None else KINECT_Z)
            )

        # For all y coords:
        else:
            row[i] = str(float(row[i]) * np.cos(ang_rad) - z * np.sin(ang_rad))

    return row


def static_kinect(row):
    # NOTE: the static skeleton has its lower back on the x=0 plane and its left foot on the z=0 plane
    # Lower back x: row[2], feet z: row[39], row[51]
    x_abs = float(row[2])
    y_abs = float(row[13])
    z_abs = min(float(row[39]), float(row[51]))
    for i in range(2, len(row) - 1):
        # For all x coords
        if i % 3 == 2:
            row[i] = str(float(row[i]) - x_abs)

        # For all y coords
        elif i % 3 == 1:
            row[i] = str(float(row[i]) - y_abs)

        # For all z coords:
        else:
            row[i] = str(float(row[i]) - z_abs)

    return row


def preprocess_dataset(runall = True, exp = ""):

    experiments_directory = f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}"
    print("Preprocessing:")

    mars_dir = f"{const.P_PREPROCESS_PATH}/mars/"   
    kinect_dir = f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}"
    if os.path.exists(mars_dir):
        shutil.rmtree(mars_dir)
    if os.path.exists(kinect_dir):
        shutil.rmtree(kinect_dir)
    os.makedirs(mars_dir)
    os.makedirs(kinect_dir)

    for experiment in tqdm(os.listdir(experiments_directory)): #Lists experiments A1, A2...
        if not runall and experiment != exp:
            continue

        print(f"Preprocessing {experiment}")
        frame_pairs = pair(experiment)

        input_dir = os.path.join(f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}", experiment)
        output_dir = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}/{experiment}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        data_buffer = pd.DataFrame()
        frames_in_cur_file = 0
        cur_file_index = 1
        cur_file = os.path.join(output_dir, f"{cur_file_index}.csv")
        # centroids = []

        trackbuffer = TrackBuffer()
        batch = BatchedData()
        sensor_data = OfflineManager(input_dir)

        first_iter = True
        invalid_frames = []

        mars_data_buffer = pd.DataFrame()

        # 
        centroid_dict = filter_kinect_frames(frame_pairs, invalid_frames, experiment)
        # print(centroid_dict)

        while not sensor_data.is_finished():
            valid_frame = False
            dataOk, framenum, detObj = sensor_data.get_data()

            if any(framenum == pair[0] for pair in frame_pairs):
                if dataOk:
                    if first_iter:
                        trackbuffer.dt = 0.1
                        first_iter = False
                    else:
                        trackbuffer.dt = detObj["posix"][0] / 1000 - trackbuffer.t

                    trackbuffer.t = detObj["posix"][0] / 1000
                    effective_data = normalize_data(detObj) #Effective_data: (n x 8) ndarray, n -> number of point clouds for a frame

                    if effective_data.shape[0] != 0:
                        trackbuffer.track(effective_data, batch)

                        if len(trackbuffer.effective_tracks) > 0:
                            track_points = trackbuffer.effective_tracks[
                                0
                            ].batch.effective_data

                            if (
                                trackbuffer.effective_tracks[0].lifetime == 0
                                and len(track_points) > 0
                            ):
                                valid_frame = True

                                chosen_track_idx = 0
                                if len(trackbuffer.effective_tracks) > 1:
                                    gt_frame = [p[1] for p in frame_pairs if framenum == p[0]][0]
                                    gt_centroid = centroid_dict[str(gt_frame)]
                                    min_centroid_dist = math.inf
                                    for track_idx in range(len(trackbuffer.effective_tracks)):
                                        dist = np.linalg.norm(trackbuffer.effective_tracks[track_idx].cluster.centroid[:3] - np.array(gt_centroid))
                                        if dist < min_centroid_dist:
                                            chosen_track_idx = track_idx
                                # if chosen_track_idx =??= 0:
                                if chosen_track_idx == 0 and len(trackbuffer.effective_tracks) > 1:
                                    print(chosen_track_idx)


                                frames_to_process = list(
                                    trackbuffer.effective_tracks[chosen_track_idx].batch.buffer
                                )


                                if RELATIVE_ENABLED:
                                    frames_to_process = relative_coordinates(
                                        frames_to_process,
                                        trackbuffer.effective_tracks[
                                            0
                                        ].cluster.centroid,
                                    )

                                    # TODO: Add relative coordinates for MARS as well
                                    

                                final_frames = format_batched_frames(frames_to_process)
                                # print(f"ASTERIOS Shape: {np.shape(final_frames)}")
                                # Save effective data in a .csv
                                data = {
                                    "Frame": framenum,
                                    "X": final_frames[:, 0],
                                    "Y": final_frames[:, 1],
                                    "Z": final_frames[:, 2],
                                    "Doppler": final_frames[:, 3],
                                    "Intensity": final_frames[:, 4],
                                }


                                # MARS Data Preprocessing
                                # Since we already have normalized data, we can just push it to format batched frames straight away
                                mars_frames = format_mars_frames(effective_data)
                                # print(f"MARS Shape : {np.shape(mars_frames)}")
                                mars_data = {
                                    "Frame": framenum,
                                    "X": mars_frames[:, 0],
                                    "Y": mars_frames[: , 1],
                                    "Z": mars_frames[:, 2],
                                    "Doppler": mars_frames[:, 3],
                                    "Intensity": mars_frames[:, 4]
                                }
                                mars_data_buffer = pd.concat(
                                    [mars_data_buffer, pd.DataFrame(mars_data)], ignore_index = True
                                )

                                # Store data in the data path
                                df = pd.DataFrame(data)
                                data_buffer = pd.concat(
                                    [data_buffer, df], ignore_index=True
                                )
                                frames_in_cur_file += 1

                                # Check if buffer size or file size limit is reached
                                if (
                                    len(data_buffer) >= const.FB_WRITE_BUFFER_SIZE
                                    or frames_in_cur_file
                                    >= const.FB_EXPERIMENT_FILE_SIZE
                                ):
                                    # Write data to CSV
                                    df = pd.DataFrame(data_buffer)
                                    df.to_csv(
                                        cur_file, mode="a", index=False, header=False
                                    )
                                    data_buffer.drop(data_buffer.index, inplace=True)

                                    # Update file index and file path if necessary
                                    if (
                                        frames_in_cur_file
                                        >= const.FB_EXPERIMENT_FILE_SIZE
                                    ):
                                        frames_in_cur_file = 0
                                        cur_file_index += 1
                                        cur_file = os.path.join(
                                            output_dir, f"{cur_file_index}.csv"
                                        )

                else:
                    batch.pop_frame()

            if not valid_frame:
                invalid_frames.append(framenum)

        # Write remaining data to CSV
        df = pd.DataFrame(data_buffer)
        df.to_csv(cur_file, mode="a", index=False, header=False)

        # np.save(f"./centroids_final/{experiment}_centroid.npy", np.array(centroids))

        mars_data_buffer.to_csv(os.path.join(const.P_PREPROCESS_PATH, "mars", f"{experiment}.csv"), mode="w", index=False, header=False)

        filter_kinect_frames(frame_pairs, invalid_frames, experiment)


def format_mmwave_to_npy(
    mode, index, mean=const.INTENSITY_MU, std_dev=const.INTENSITY_STD
):

    # Exactly how many frames we process
    BATCH_SIZE = 3
    FUSE = False

    experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}{mode}/"
    output_path = f"{const.P_FORMATTED_PATH}{const.P_MMWAVE_DIR}{index}/"
    main_list = []

    experiments = os.listdir(experiments_directory)
    # experiments_sorted = sorted(experiments, key=extract_parts)
    experiments_sorted = experiments

    for experiment in experiments_sorted:
        # print(f"doing {experiment}")
        experiment_path = os.path.join(experiments_directory, experiment)
        filenames = os.listdir(experiment_path)
        filenames_sorted = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
        for filename in filenames_sorted:
            with open(os.path.join(experiment_path, filename), "r") as file:
                reader = csv.reader(file)
                rows = list(reader)

                current_frame = None
                frame_array = []
                for row in rows:
                    if current_frame is None:
                        current_frame = int(row[0])
                        frame_array.append([float(i) for i in row[1:6]])

                    elif current_frame == int(row[0]):
                        frame_array.append([float(i) for i in row[1:6]])

                    else:
                        main_list.append(
                            format_single_frame_mode(
                                np.array(frame_array, dtype=np.float32),
                                mean,
                                std_dev,
                                BATCH_SIZE,
                                FUSE,
                            )
                        )

                        frame_array = []
                        current_frame = int(row[0])
                        frame_array.append([float(i) for i in row[1:6]])
                if len(frame_array) > 0:
                    main_list.append(
                        format_single_frame_mode(
                            np.array(frame_array, dtype=np.float32),
                            mean,
                            std_dev,
                            BATCH_SIZE,
                            FUSE,
                        )
                )
    print(np.array(main_list).shape)
    # Save to output .npy file
    np.save(
        os.path.join(output_path, f"{mode}_mmWave.npy"),
        np.array(main_list),
    )


def format_kinect_to_npy(mode, index):
    experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}{mode}/"
    experiments = os.listdir(experiments_directory)
    # experiments_sorted = sorted(experiments, key=extract_parts)
    experiments_sorted = experiments
    output_path = f"{const.P_FORMATTED_PATH}{const.P_KINECT_DIR}{index}/"

    main_list = []
    for experiment in experiments_sorted:
        with open(os.path.join(experiments_directory, experiment), "r") as exp:

            frames = pd.read_csv(exp, header=None)
            for _, frame in frames.iterrows():
                main_list.append(np.array(frame[2:59]).reshape(-1, 3).T.flatten())

    print(np.array(main_list).shape)
    # Save to output .npy file
    np.save(
        os.path.join(output_path, f"{mode}_labels.npy"),
        np.array(main_list),
    )


def format_dataset(index):
    sets = ["training", "validate", "testing"]

    with keep.presenting():
        os.mkdir(f"{const.P_FORMATTED_PATH}{const.P_MMWAVE_DIR}{index}/")
        os.mkdir(f"{const.P_FORMATTED_PATH}{const.P_KINECT_DIR}{index}/")
        for set_mode in sets:
            # Preprocess .csvs into numpy arrays and save them in one file
            format_mmwave_to_npy(set_mode, index)
            format_kinect_to_npy(set_mode, index)


def extract_parts(filename):
    base, ext = os.path.splitext(filename)
    numeric_part = "".join(filter(str.isdigit, base))
    alpha_part = "".join(filter(str.isalpha, base))
    return int(numeric_part), alpha_part, ext


def split_sets(prefixes):

    kinect_directory = f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}"
    mmwave_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}"

    directories = [kinect_directory, mmwave_directory]

    for directory in directories:
        try:
            shutil.rmtree(os.path.join(f"{directory}", "training"))
        except Exception as e:
            pass
        try:
            shutil.rmtree(os.path.join(f"{directory}", "validate"))
        except Exception as e:
            pass
        try:
            shutil.rmtree(os.path.join(f"{directory}", "testing"))
        except Exception as e:
            pass
        
    # validate_prefix, testing_prefix = random_split_sets()
    validate_prefix = prefixes[0]
    testing_prefix = prefixes[1]

    for directory in directories:
        experiments = os.listdir(directory)

        os.makedirs(os.path.join(f"{directory}", "training"))
        os.makedirs(os.path.join(f"{directory}", "validate"))
        os.makedirs(os.path.join(f"{directory}", "testing"))

        for experiment in experiments:
            if (
                experiment.find("training") == -1
                and experiment.find("validate") == -1
                and experiment.find("testing") == -1
            ):
                source = f"{directory}/{experiment}"
                if os.path.isdir(source):
                    if any(experiment.find(prefix) != -1 for prefix in validate_prefix):
                        shutil.copytree(
                            source,
                            os.path.join(f"{directory}", "validate", f"{experiment}")
                            # f"{directory}/validate/{experiment}",
                        )

                    elif any(
                        experiment.find(prefix) != -1 for prefix in testing_prefix
                    ):
                        shutil.copytree(
                            source,
                            os.path.join(f"{directory}", "testing", f"{experiment}")
                            # f"{directory}/testing/{experiment}",
                        )
                    else:
                        shutil.copytree(
                            source,
                            os.path.join(f"{directory}", "training", f"{experiment}")
                            # f"{directory}/training/{experiment}",
                        )
                else:
                    if any(experiment.find(prefix) != -1 for prefix in validate_prefix):
                        shutil.copy(
                            source,
                            os.path.join(f"{directory}", "validate", f"{experiment}")
                            # f"{directory}/validate/{experiment}",
                        )

                    elif any(
                        experiment.find(prefix) != -1 for prefix in testing_prefix
                    ):
                        shutil.copy(
                            source,
                            os.path.join(f"{directory}", "testing", f"{experiment}")
                            # f"{directory}/testing/{experiment}",
                        )
                    else:
                        shutil.copy(
                            source,
                            os.path.join(f"{directory}", "training", f"{experiment}")
                            # f"{directory}/training/{experiment}",
                        )


##################################################################

# Run the preprocessing and formatting steps for the raw dataset.

##################################################################

# 10 randomly split sets [validation, testing]
# sets = [
#     [["A6", "A4", "B2"], ["B1", "B4", "A5"]],
#     [["B1", "B3", "A3"], ["B2", "B8", "B7"]],
#     [["B4", "A2", "B7"], ["B2", "B5", "A3"]],
#     [["A6", "A4", "B9"], ["B2", "B6", "A5"]],
#     [["B5", "B2", "A7"], ["B4", "A2", "A3"]],
#     [["B6", "B9", "B5"], ["A7", "A6", "B3"]],
#     [["A6", "B8", "A3"], ["B3", "B4", "A4"]],
#     [["B1", "A7", "B8"], ["A5", "A3", "B6"]],
#     [["B2", "B5", "B8"], ["A7", "B6", "A6"]],
#     [["A4", "B6", "A7"], ["B2", "B5", "B1"]],
# ]

# print("Preprocessing:")
preprocess_dataset()


def format_experiment(experiment):
    mean=const.INTENSITY_MU 
    std_dev=const.INTENSITY_STD
    BATCH_SIZE = 3
    FUSE = False

    main_list = []
    
    kinect_path = os.path.join(const.P_PREPROCESS_PATH, "kinect", f"{experiment}.csv")
    mmwave_path = os.path.join(const.P_PREPROCESS_PATH, "mmWave", experiment)
    mars_path = os.path.join(const.P_PREPROCESS_PATH, "mars", f"{experiment}.csv")
    filenames = os.listdir(mmwave_path)
    filenames_sorted = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    # print(mmdirs)
    for idx in filenames_sorted:
        with open(os.path.join(mmwave_path, idx), "r") as file:
                reader = csv.reader(file)
                rows = list(reader)

                current_frame = None
                frame_array = []
                for row in rows:
                    if current_frame is None:
                        current_frame = int(row[0])
                        frame_array.append([float(i) for i in row[1:6]])

                    elif current_frame == int(row[0]):
                        frame_array.append([float(i) for i in row[1:6]])

                    else:
                        main_list.append(
                            format_single_frame_mode(
                                np.array(frame_array, dtype=np.float32),
                                mean,
                                std_dev,
                                BATCH_SIZE,
                                FUSE,
                            )
                        )

                        frame_array = []
                        current_frame = int(row[0])
                        frame_array.append([float(i) for i in row[1:6]])
                if len(frame_array) > 0:
                    main_list.append(
                        format_single_frame_mode(
                            np.array(frame_array, dtype=np.float32),
                            mean,
                            std_dev,
                            BATCH_SIZE,
                            FUSE,
                        )
                )   
    print(np.array(main_list).shape)

    kinect_data = []
    with open(kinect_path, "r") as exp:
            frames = pd.read_csv(exp, header=None)
            for _, frame in frames.iterrows():
                kinect_data.append(np.array(frame[2:59]).reshape(-1, 3).T.flatten())
    print(np.array(kinect_data).shape)



    # MARS 
    mars_data = []
    with open(mars_path, "r") as file:
        reader = csv.reader(file)
        rows = list(reader)

        current_frame = None
        frame_array = []
        for row in rows:
            if current_frame is None:
                current_frame = int(row[0])
                frame_array.append([float(i) for i in row[1:6]])

            elif current_frame == int(row[0]):
                frame_array.append([float(i) for i in row[1:6]])

            else:
                mars_data.append(
                    format_single_frame_mode(
                        np.array(frame_array, dtype=np.float32),
                        mean,
                        std_dev,
                        1,
                        FUSE,
                    )
                )

                frame_array = []
                current_frame = int(row[0])
                frame_array.append([float(i) for i in row[1:6]])
        if len(frame_array) > 0:
            mars_data.append(
                format_single_frame_mode(
                    np.array(frame_array, dtype=np.float32),
                    mean,
                    std_dev,
                    1,
                    FUSE,
                )
        )  
    print(np.array(mars_data).shape) 

    exp_formatted_path = os.path.join(const.P_DATA_PATH, "expformatted")
    os.makedirs(exp_formatted_path, exist_ok=True)
    os.makedirs(os.path.join(exp_formatted_path, "mmWave"), exist_ok=True)
    os.makedirs(os.path.join(exp_formatted_path, "kinect"), exist_ok=True)
    os.makedirs(os.path.join(exp_formatted_path, "mars"), exist_ok=True)
    np.save(
        os.path.join(exp_formatted_path, "mmWave", f"{experiment}.npy"),
        np.array(main_list),
    )
    np.save(
        os.path.join(exp_formatted_path, "kinect", f"{experiment}.npy"),
        np.array(kinect_data),
    )
    np.save(
        os.path.join(exp_formatted_path, "mars", f"{experiment}.npy"),
        np.array(mars_data)
    )


def run_experiment_formatting():
    experiments = os.listdir(os.path.join(const.P_PREPROCESS_PATH, "mmWave"))
    for exp in experiments:

        print(f"Running formatting for {exp}")
        format_experiment(exp)


run_experiment_formatting()