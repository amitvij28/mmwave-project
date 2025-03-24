import math
import sys
import os
import copy
import pandas as pd
import numpy as np
import warnings
import json
from sklearn import metrics
import constants as const
import tensorflow as tf

# DISABLE PANDAS WARNINGS
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppresses INFO and WARNING messages
tf.get_logger().setLevel("ERROR")  # Suppresses other logs



from PyQt5.QtWidgets import QApplication
from wakepy import keep

from Visualizer import VisualManager
from Utils import OfflineManager, format_single_frame, normalize_data
from keras.models import load_model
from Tracking import (
    TrackBuffer,
    BatchedData,
)
from preprocessing import pair, translate_kinect

########### Set the constants here ############
# EXPERIMENT = "B1"

# GROUND_TRUTH_PATH = f"{const.P_LOG_PATH}{const.P_KINECT_DIR}{EXPERIMENT}.csv"
# MMWAVE_PATH = f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}{EXPERIMENT}"

MARS_MODEL = f"./model/MARS.h5"
ASTERIOS_MODEL = f"./model/trained_model.h5"

JSON_OUTPUT_PATH = f"{const.P_DATA_PATH}/sim/"

NO_OF_HUMANS = 1
#####################################################

def get_mmwave_path(experiment):
    return f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}{experiment}"

def get_kinect_path(experiment):
    return f"{const.P_LOG_PATH}{const.P_KINECT_DIR}{experiment}.csv"

# Store as {'ast':{'data':[]}, 'mars':{'data':[]}, 'error':{}, 'gt':{}}
def store_json(data, experiment):
    main_data = {'error':data['error'] if 'error' in data else None}
    
    gt_data = []
    mars_data = []
    ast_data = []

    for gt in data['gt']:
        gt_data.append(gt.reshape(3,-1).astype(float).tolist()) 
    
    for mars in data['mars']:
        reshaped_mars = mars.reshape(3,-1)
        mars_data.append(np.vstack([reshaped_mars[0], reshaped_mars[2], reshaped_mars[1]]).tolist())
    
    for ast in data['ast']:
        tracks = []
        for track in ast:
            reshaped_data = track.keypoints.reshape(3,-1)
            # tracks.append({"keypoint":reshaped_data.tolist(), "state": [track.state.x[0][0], track.state.x[1][0]]})
            reshaped_data[0] *= -1
            reshaped_data[0] += track.state.x[0]
            reshaped_data[2] += track.state.x[1]
            tracks.append(reshaped_data.tolist())
        ast_data.append(tracks)


    main_data['gt'] = gt_data
    main_data['mars'] = mars_data
    main_data['ast'] = ast_data
    
    with open(f"{JSON_OUTPUT_PATH}{experiment}.json", "w") as f:
        f.write(json.dumps(main_data))
    print(f"Data stored at {JSON_OUTPUT_PATH}{experiment}")


def calc_mean_error(ground_truth, prediction):
    error = {}
    error['x'] = metrics.mean_absolute_error(ground_truth[0], prediction[0])
    error['y'] = metrics.mean_absolute_error(ground_truth[2], prediction[2])
    error['z'] = metrics.mean_absolute_error(ground_truth[1], prediction[1])
    return error

# TODO: Calculate errors for body regions
def error_calculation(gt_data, ast_data, mars_data):
    
    # Find the difference in number of tracked objects/humans
    ground_truth = gt_data.reshape(3, -1).astype(float)
    mars_prediction = mars_data.reshape(3, -1)
    mars_prediction = np.vstack([mars_prediction[0], mars_prediction[2], mars_prediction[1]])
    tracks = []
    track_centroids = []
    chosen_track = None
    for track in ast_data:
        reshaped_data = track.keypoints.reshape(3,-1)
        reshaped_data[0] *= -1
        reshaped_data[0] += track.state.x[0]
        reshaped_data[2] += track.state.x[1]
        track_centroids.append(np.mean(reshaped_data, axis=1))
        tracks.append(reshaped_data)
    
    min_index = 0
    if len(tracks) == NO_OF_HUMANS:
        chosen_track = tracks[0]
    elif len(tracks) == 0:
        return {"ast":{"x":0,"y":0,"z":0},"mars":{"x":0,"y":0,"z":0}}
    else:
        gt_centroid = np.mean(ground_truth, axis=1)
        # print(gt_centroid, track_centroids)
        min_diff = math.inf
        for index in range(len(track_centroids)):
            diff = np.linalg.norm(track_centroids[index] - gt_centroid)
            if diff < min_diff:
                min_diff = diff
                min_index = index
        chosen_track = tracks[min_index]

    # TODO: Update localisation of mars based upon the chosen track
    mars_prediction[0] *= -1
    mars_prediction[0] += ast_data[min_index].state.x[0]
    mars_prediction[2] += ast_data[min_index].state.x[1]

    ast_error = calc_mean_error(ground_truth, chosen_track)
    mars_error = calc_mean_error(ground_truth, mars_prediction)


    # Calculate error based upon locations of joints (arms, legs, head)
    return {"ast":ast_error, "mars": mars_error}


def transform_ground_truth(data: pd.DataFrame, kinect_x, kinect_z):
    transformed_data = []
    for i, row in data.iterrows():
        transformed_data.append( translate_kinect(row, kinect_x, kinect_z))
    return pd.DataFrame(transformed_data)


def reshape_groundtruth(frame):
    return np.array(frame[2:59]).reshape(-1, 3).T.flatten()


def offline_main(experiment):
    mmwave_path = get_mmwave_path(experiment)
    kinect_path = get_kinect_path(experiment)
    if not os.path.exists(mmwave_path):
        raise ValueError(f"No mmwave file found in the path: {mmwave_path}")

    if not os.path.exists(kinect_path):
        raise ValueError(f"No ground truth file found in the path: {kinect_path}")

    frame_pairs = pair(experiment) #[(mmwave_frame_no, kinect_frame_no),...]
    kinect_frames = [i[1] for i in frame_pairs ]
    mmwave_frames = [i[0] for i in frame_pairs]
    print(f"Simulation length: {len(frame_pairs)}")
    ground_truth_data = pd.read_csv(kinect_path, header=None)

    # ground_truth_data = ground_truth_data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    # print(frame_pairs)

    sensor_data = OfflineManager(mmwave_path, mmwave_frames)
    # Filter ground truth frames based upon the frame pairs
    # print(ground_truth_data)
    filtered_ground_truth = ground_truth_data[ground_truth_data.iloc[:, 1].isin(kinect_frames)]
    
    kinect_x = -0.1
    if experiment[0] == "A":
        kinect_x = 0.8
    kinect_z = 0.2
    if experiment[0] == "A":
        kinect_z = 0.1
         
    ground_truth = transform_ground_truth(filtered_ground_truth, kinect_x, kinect_z)
    ground_truth = ground_truth.drop(ground_truth.columns[[20,21,22, 32,33,34]], axis=1)  # Removes first and third columns


    SLEEPTIME = 0.1  # from radar config "frameCfg"


    # app = QApplication(sys.argv)
    # visual = VisualManager()
    trackbuffer = TrackBuffer()
    batch = BatchedData()
    first_iter = True


    asterios_model = load_model(ASTERIOS_MODEL)
    mars_model = load_model(MARS_MODEL)


    main_data = {"ast":[], "gt":[], "mars":[], "error":[]}

    while not sensor_data.is_finished():
        try:
            dataOk, framenum, detObj = sensor_data.get_data()
            
            if dataOk:

                # TODO: Match Kinect frame with the frame num
                
                gt_framenum = kinect_frames[mmwave_frames.index(framenum)]
                # print(f"gt frame: {gt_framenum}")
                # print(f"ggg: {ground_truth.iloc[:,1].tolist()}")
                kinect_row = ground_truth.loc[ground_truth.iloc[:, 1] == gt_framenum].iloc[0]
                kinect_array = reshape_groundtruth(kinect_row)
                
                # ---------- ASTERIOS MODEL----------------
                if first_iter:
                    trackbuffer.dt = SLEEPTIME
                    first_iter = False
                else:
                    trackbuffer.dt = detObj["posix"][0] / 1000 - trackbuffer.t

                trackbuffer.t = detObj["posix"][0] / 1000
                # Apply scene constraints, point translation and axis normalization
                effective_data = normalize_data(detObj)

                if effective_data.shape[0] != 0:
                    # Tracking module
                    trackbuffer.track(effective_data, batch)

                    # Posture Estimation module
                    trackbuffer.estimate_posture(asterios_model)

                #TODO:--------------- MARS MODEL---------------
                mars_data = effective_data.reshape(1, *effective_data.shape)
                mars_data = format_single_frame(mars_data, is_mars=True)
                mars_prediction = mars_model.predict(mars_data, verbose=0)
                
                #TODO:--------------- mmMesh Model-------------




                # TODO: -------------Error Calc---------------
                # error_data = error_calculation(kinect_array, trackbuffer.effective_tracks, mars_prediction)

                main_data['ast'].append(copy.deepcopy(trackbuffer.effective_tracks))
                main_data['gt'].append(kinect_array)
                main_data['mars'].append(mars_prediction)
                # main_data['error'].append(error_data)
                # visual.update(trackbuffer, detObj, kinect_array, mars_prediction, error_data)
            
        except KeyboardInterrupt:
            break


    store_json(main_data, experiment)


# IGNORE_EXPERIMENTS = ['A1','A2','B1','B2','C1','D1','T1', 'G1']
def main():
    experiments = os.listdir(f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}")
    print(experiments)
    for experiment in experiments:
        if experiment[0] != "C":
        # if experiment:
            continue
        print(f"Running {experiment} simulation")
        offline_main(experiment)

main()

