import sys
import os
import copy
import pandas as pd
import numpy as np
import warnings
import json
from sklearn import metrics
import constants as const

# DISABLE PANDAS WARNINGS
warnings.simplefilter(action='ignore', category=FutureWarning)



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
EXPERIMENT = "A2"

GROUND_TRUTH_PATH = f"./dataset/log/kinect/{EXPERIMENT}.csv"
MMWAVE_PATH = f"./dataset/log/mmWave/{EXPERIMENT}"

MARS_MODEL = f"./model/MARS.h5"
ASTERIOS_MODEL = f"./model/trained_model.h5"

JSON_OUTPUT_PATH = f"{const.P_DATA_PATH}/analysis/"

NO_OF_HUMANS = 1
#####################################################

# Store as {'ast':{'data':[]}, 'mars':{'data':[]}, 'error':{}, 'gt':{}}
def store_json(data):
    main_data = {'error':data['error']}
    
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
            # TODO: check for the mirror skeleton part to see if it's effecting the error   
            
            reshaped_data = track.keypoints.reshape(3,-1)
            
            reshaped_data[0] *= -1
            reshaped_data[0] += track.state.x[0]
            reshaped_data[2] += track.state.x[1]
            tracks.append(reshaped_data.tolist())
        ast_data.append(tracks)


    main_data['gt'] = gt_data
    main_data['mars'] = mars_data
    main_data['ast'] = ast_data
    
    with open(f"{JSON_OUTPUT_PATH}{EXPERIMENT}.json", "w") as f:
        f.write(json.dumps(main_data))
    print(f"Data stored at {JSON_OUTPUT_PATH}{EXPERIMENT}")


def calc_mean_error(ground_truth, prediction):
    error = {}
    error['x'] = metrics.mean_absolute_error(ground_truth[0], prediction[0])
    error['y'] = metrics.mean_absolute_error(ground_truth[2], prediction[2])
    error['z'] = metrics.mean_absolute_error(ground_truth[1], prediction[1])
    return error

def error_calculation(gt_data, ast_data, mars_data):
    
    # Find the difference in number of tracked objects/humans
    ground_truth = gt_data.reshape(3, -1)
    mars_prediction = mars_data.reshape(3, -1)
    mars_prediction = np.vstack([mars_prediction[0], mars_prediction[2], mars_prediction[1]])
    tracks = []
    chosen_track = None
    for track in ast_data:
        reshaped_data = track.keypoints.reshape(3,-1)
        reshaped_data[0] *= -1
        reshaped_data[0] += track.state.x[0]
        reshaped_data[2] += track.state.x[1]
        tracks.append(reshaped_data)
    
    if len(tracks) == NO_OF_HUMANS:
        chosen_track = tracks[0]

    # print(f"gt: {np.shape(ground_truth)} {np.shape(chosen_track)}")

    # Calculate error based upon the mean 19 joints
   
    ast_error = calc_mean_error(ground_truth, chosen_track)
    mars_error = calc_mean_error(ground_truth, mars_prediction)


    # Calculate error based upon locations of joints (arms, legs, head)
    return {"ast":ast_error, "mars": mars_error}


def transform_ground_truth(data: pd.DataFrame):
    transformed_data = []
    for i, row in data.iterrows():
        transformed_data.append( translate_kinect(row))
    return pd.DataFrame(transformed_data)


def reshape_groundtruth(frame):
    return np.array(frame[2:59]).reshape(-1, 3).T.flatten()


def offline_main():
    if not os.path.exists(MMWAVE_PATH):
        raise ValueError(f"No mmwave file found in the path: {MMWAVE_PATH}")

    if not os.path.exists(GROUND_TRUTH_PATH):
        raise ValueError(f"No ground truth file found in the path: {GROUND_TRUTH_PATH}")

    frame_pairs = pair(EXPERIMENT) #[(mmwave_frame_no, kinect_frame_no),...]
    kinect_frames = [i[1] for i in frame_pairs ]
    mmwave_frames = [i[0] for i in frame_pairs]


    ground_truth_data = pd.read_csv(GROUND_TRUTH_PATH)

    # ground_truth_data = ground_truth_data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))


    sensor_data = OfflineManager(MMWAVE_PATH, mmwave_frames)

    # Filter ground truth frames based upon the frame pairs
    filtered_ground_truth = ground_truth_data[ground_truth_data.iloc[:, 1].isin(kinect_frames)]
    ground_truth = transform_ground_truth(filtered_ground_truth)
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
                kinect_row = ground_truth.loc[ground_truth.iloc[:, 1] == gt_framenum].iloc[0]
                kinect_array = reshape_groundtruth(kinect_row)
                # print(np.shape(kinect_array.reshape(3, -1)))

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
                mars_prediction = mars_model.predict(mars_data)
                
                #TODO:--------------- mmMesh Model-------------




                # TODO: -------------Error Calc---------------
                error_data = error_calculation(kinect_array, trackbuffer.effective_tracks, mars_prediction)

                # Store the avg error rate in csv
                main_data['ast'].append(copy.deepcopy(trackbuffer.effective_tracks))
                main_data['gt'].append(kinect_array)
                main_data['mars'].append(mars_prediction)
                main_data['error'].append(error_data)

                
                # visual.update(trackbuffer, detObj, kinect_array, mars_prediction, error_data)
            
        except KeyboardInterrupt:
            break


    store_json(main_data)

offline_main()

