import matplotlib.pyplot as plt
# from networkx import draw
import numpy as np
import json
import pandas as pd
import os
import constants as const
from matplotlib.widgets import Slider
from keras.models import Model, load_model


DATASET = "dataset_main"

KINECT_FORMATTED_PATH = os.path.join(".", DATASET, "expformatted", "kinect")
MMWAVE_FORMATTED_PATH = os.path.join(".", DATASET, "expformatted", "mmWave")
MARS_FORMATTED_PATH = os.path.join(".", DATASET, "expformatted", "mars")

MODEL_PATH = os.path.join(".", "model", "main_tests")
# KINECT_LOG_PATH = os.path.join(".", DATASET, "log", "kinect")
# MMWAVE_LOG_PATH = os.path.join(".", DATASET, "log", "mmWave")
# MMWAVE_PREPROCESS_PATH = os.path.join(".", DATASET, "preprocessed", "mmWave")
# KINECT_PREPROCESS_PATH = os.path.join(".", DATASET, "preprocessed", "kinect")

CONNECTIONS = [
            (0, 1),  # SpineBase to SpineMid
            (1, 18),  # SpineMid to SpineShoulder
            (2, 3),  # Neck to Head
            (18, 4),  # SpineShoulder to ShoulderLeft
            (18, 7),  # SpineShoulder to ShoulderRight
            (4, 5),  # ShoulderLeft to ElbowLeft
            (5, 6),  # ElbowLeft to WristLeft
            (7, 8),  # ShoulderRight to ElbowRight
            (8, 9),  # ElbowRight to WristRight
            (0, 14),  # SpineBase to HipRight
            (14, 15),  # HipRight to KneeRight
            (15, 16),  # KneeRight to AnkleRight
            (16, 17),  # AnkleRight to FootRight
            (0, 10),  # SpineBase to HipLeft
            (10, 11),  # HipLeft to KneeLeft
            (11, 12),  # KneeLeft to AnkleLeft
            (12, 13),  # AnkleLeft to FootLeft
            (2, 18),  # Neck to SpineShoulder
]

KEYPOINT_COLOURS = [
            "blue",  # SpineBase,
            "blue",  # SpineMid,
            "blue",  # Neck,
            "black",  # Head,
            "green",  # ShoulderLeft,
            "red",  # ElbowLeft,
            "green",  # WristLeft,
            "green",  # ShoulderRight,
            "red",  # ElbowRight,
            "green",  # WristRight,
            "blue",  # HipLeft,
            "green",  # KneeLeft,
            "green",  # AnkleLeft,
            "green",  # FootLeft,
            "blue",  # HipRight,
            "green",  # KneeRight,
            "green",  # AnkleRight,
            "green",  # FootRight,
            "blue",  # SpineShoulder
]

def draw_skeleton(track, ax, isgroundtruth=False):
    track =track.reshape(3, -1)
    # Translate Y axis
    track[2] += 2
    # ax.clear()

    setup_3D_subplot(ax)
    for connection in CONNECTIONS:
        keypoint_1 = connection[0]
        keypoint_2 = connection[1]

        x_values = [float(track[0][keypoint_1]), float(track[0][keypoint_2])]
        y_values = [float(track[2][keypoint_1]), float(track[2][keypoint_2])]
        z_values = [float(track[1][keypoint_1]), float(track[1][keypoint_2])]
        ax.plot(x_values, y_values, z_values, color="grey")

    for keypoint_index in range(len(track[0])):
        color = KEYPOINT_COLOURS[keypoint_index] if not isgroundtruth else "grey"
        marker = (
            "o" if keypoint_index != 3 else "s"
        )  
        ax.scatter(
            float(track[0][keypoint_index]),
            float(track[2][keypoint_index]),
            float(track[1][keypoint_index]),
            c=color,
            marker=marker,
            s=(10 if keypoint_index == 3 else 10) if not isgroundtruth else 3, 
        )
    track[2] -= 2

def setup_3D_subplot(subplot):
    subplot.clear()
    axis_dim = const.V_3D_AXIS
    subplot.set_xlim(-1.5, 1.5)
    subplot.set_ylim(0, 2)
    subplot.set_zlim(0, 2)
    subplot.set_xlabel("X")
    subplot.set_ylabel("Z")
    subplot.set_zlabel("Y")
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_zticks([])
    subplot.set_xticklabels([])
    subplot.set_yticklabels([])
    subplot.set_zticklabels([])
    subplot.invert_yaxis()
    subplot.invert_xaxis()
    return subplot.scatter([], [], [])


# def load_mmwave(experiment):
#     mmwave_dir = os.path.join(MMWAVE_PREPROCESS_PATH, experiment)
#     files = os.listdir(mmwave_dir)
#     sorted_files = sorted(files, key= lambda k: int(k.split(".")[0]))
#     mmwave_data = []
#     for f in sorted_files:  
#         data = pd.read_csv(os.path.join(mmwave_dir, f), header=None)
#         data = np.array(data).tolist()
#         mmwave_data.extend(data)
#     return np.array(mmwave_data)


# def load_kinect(experiment):
#     kinect_file = os.path.join(KINECT_PREPROCESS_PATH, f"{experiment}.csv")
#     kinect_data = pd.read_csv(kinect_file, header=None)
#     return np.array(kinect_data)


def update_baseline_cloud(data, ax):
    data = data[0]
    ax.clear()
    setup_3D_subplot(ax)
    x_values = data[:,:, 0].flatten()
    y_values = data[:,:, 1].flatten()
    z_values = data[:,:, 2].flatten()

    intensities = data[:, :, 3].flatten()
    
    non_zero_mask = (x_values != 0) | (y_values != 0) | (z_values != 0)

    # Apply the mask to all data arrays
    x_filtered = x_values[non_zero_mask]
    y_filtered = y_values[non_zero_mask]
    z_filtered = z_values[non_zero_mask]
    intensities_filtered = intensities[non_zero_mask]


    # y_values 
    ax.scatter(x_filtered, y_filtered, z_filtered, c=intensities_filtered, cmap='Reds', s=30, alpha=0.8,
                          edgecolors='gray', linewidth=0.8)

def update_temporal_cloud(data, ax):
    ax.clear()
    setup_3D_subplot(ax)
    x_values = data[:, :, :, 0].flatten()
    y_values = data[:, :, :, 1].flatten()
    z_values = data[:, :, :, 2].flatten()

    intensities = data[:, :,:, 3].flatten()
    
    non_zero_mask = (x_values != 0) | (y_values != 0) | (z_values != 0)

    # Apply the mask to all data arrays
    x_filtered = x_values[non_zero_mask]
    y_filtered = y_values[non_zero_mask]
    z_filtered = z_values[non_zero_mask]
    intensities_filtered = intensities[non_zero_mask]


    # y_values 
    ax.scatter(x_filtered, y_filtered, z_filtered, c=intensities_filtered, cmap='Reds', s=30, alpha=0.8,
                          edgecolors='gray', linewidth=0.8)




def run_preprocessed(experiment, model=None):
    # mmwave_data = load_mmwave(experiment)
    # kinect_data = load_kinect(experiment)
    kinect_data = np.load(os.path.join(KINECT_FORMATTED_PATH, f"{experiment}.npy"))
    temporal_data = np.load(os.path.join(MMWAVE_FORMATTED_PATH, f"{experiment}.npy"))
    baseline_data = np.load(os.path.join(MARS_FORMATTED_PATH, f"{experiment}.npy"))
    
    temporal_model, baseline_model = None, None
    if model:
        temporal_model = load_model(os.path.join(MODEL_PATH, f"tmars_{model}.h5"))
        baseline_model = load_model(os.path.join(MODEL_PATH, f"bmars_{model}.h5"))


    current_frame = 0
    experiment_len = kinect_data.shape[0]

    # fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    # plt.subplots_adjust(bottom=0.25)

    fig = plt.figure(figsize=(14, 8))
    axs = []
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection='3d')
            axs.append(ax)

    plt.subplots_adjust(bottom=0.15)

    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_ax, 'Frame', current_frame, experiment_len, valinit=0, valstep=1)

    def update(val):
        index = int(slider.val)
        
        # baseline point cloud
        update_baseline_cloud(baseline_data[index], axs[0])
        
        # temporal point cloud
        update_temporal_cloud(temporal_data[index], axs[1])
        
        # Kinect ground truth
        
        # setup_3D_subplot(axs[3])
        draw_skeleton(kinect_data[index], axs[3], True)
        # draw_skeleton(kinect_data[index], axs[1])
        
        # Baseline model
        if baseline_model:
            predictions = baseline_model.predict(np.array([baseline_data[index]]))
            draw_skeleton(predictions, axs[4])

        # Temporal model   
        if temporal_model:
            # print(temporal_data[index].shape)
            predictions = temporal_model.predict(np.array([temporal_data[index]]))
            draw_skeleton(predictions, axs[5])




        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        current = int(slider.val)
        if event.key == 'left' and current > 0:
            slider.set_val(current - 1)
        elif event.key == 'right' and current < experiment_len:
            slider.set_val(current + 1)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()






run_preprocessed("CAFA", "A1") #623
# run_kinect_plot("AAAA")

# print(load_kinect("AAAA").shape)
