import numpy as np
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

PIXEL_TO_METERS = 0.000265


##### General Flags #####
PROFILING = False
SCREEN_CONNECTED = False

##### Paths and Ports #####
P_CONFIG_PATH = "./config_cases/our_config_8.5m.cfg"
P_MODEL_PATH = "./model/tmars_amit.h5"
P_DATA_PATH = "./dataset_asterios"

P_LOG_PATH = f"{P_DATA_PATH}/log"
P_PREPROCESS_PATH = f"{P_DATA_PATH}/preprocessed"
P_FORMATTED_PATH = f"{P_DATA_PATH}/formatted"

P_KINECT_DIR = "/kinect/"
P_MMWAVE_DIR = "/mmWave/"

P_PROFILING_PATH = "./profiling/"

P_CLI_PORT = "COM8"
P_DATA_PORT = "COM9"

###### Scene Setup ######
# Sensitive Object Coordinates
M_X = 0.32
M_Y = -0.6
M_Z = 1.3

# Smart Window Attributes
# SCREEN_SIZE = [1920 * PIXEL_TO_METERS, 1200 * PIXEL_TO_METERS]  # Laptop
SCREEN_SIZE = [1.6, 1.1]  # Smart Window Size
SCREEN_HEIGHT = 1.3  # Smart Window Installation Height

# Sensor Attributes
S_HEIGHT = 1.5 # Sensor Installation Height
S_TILT = 0  # Sensor Tilt: (-180, 180)

# Plot Parameters
V_SCALLING = 1  # Scaling parameter (only for emulating)

V_3D_AXIS = [[-2.5, 2.5], [0, 5], [0, 3]]
V_SCREEN_FADE_SIZE_MAX: float = 0.3
V_SCREEN_FADE_SIZE_MIN: float = 0.2
V_SCREEN_FADE_WEIGHT: float = (
    0.08  # square size reduction m) per 1 meter of distance from sensor
)
V_BBOX_HEIGHT = 1.8
V_BBOX_EYESIGHT_HEIGHT = 1.75


###### Experiment Logging #######
FB_FRAMES_SKIP = 0
FB_EXPERIMENT_FILE_SIZE = 200
FB_WRITE_BUFFER_SIZE = 40  # NOTE: must divide FB_EXPERIMENT_FILE_SIZE
FB_READ_BUFFER_SIZE = 40


####### Clustering #######
# Ringbuffer
FB_FRAMES_BATCH = 2
FB_FRAMES_BATCH_STATIC = 2

# DBScan
DB_Z_WEIGHT = 0.4
DB_RANGE_WEIGHT = 0.03
DB_EPS = 0.3
DB_MIN_SAMPLES_MIN = 35

# Inner DBScan (Currently inactive)
DB_POINTS_THRES = 40
DB_SPREAD_THRES = 0.7
DB_INNER_EPS = 0.1
DB_INNER_MIN_SAMPLES = 8
DB_MIN_SAMPLES_MAX = 25


###### Tracking and Kalman ######
# Tracks
TR_MAX_TRACKS = 4
TR_LIFETIME_DYNAMIC = 3  # sec
TR_LIFETIME_STATIC = 7  # sec
TR_VEL_THRES = 0.12  # Velocity threshold for STATIC or DYNAMIC track
TR_GATE = 4.5

# Kalman Noise Variances
KF_R_STD = 0.1
KF_Q_STD = 1

# Initialization values
KF_P_INIT = 0.1
KF_GROUP_DISP_EST_INIT = 0.1

# GTRACK pointnum & spread estimation
KF_ENABLE_EST = False
KF_A_N = 0.9
KF_EST_POINTNUM = 10
KF_SPREAD_LIM = [0.2, 0.2, 2, 1.2, 1.2, 0.2]
KF_A_SPR = 0.9

############### Model ####################
# Intensity Normalization
INTENSITY_MU = 27.0187
INTENSITY_STD = 70.351

MODEL_MIN_INPUT = 0
MODEL_DEFAULT_POSTURE = np.array(
    [
        0.0000,
        -0.0007,
        -0.0006,
        -0.0038,
        -0.1820,
        -0.2540,
        -0.2579,
        0.1830,
        0.2957,
        0.2940,
        -0.0805,
        -0.1141,
        -0.1232,
        -0.1358,
        0.0796,
        0.1436,
        0.1558,
        0.1720,
        -0.0007,
        0.7699,
        1.0906,
        1.4020,
        1.5513,
        1.2893,
        1.0360,
        0.7994,
        1.2865,
        1.0483,
        0.8117,
        0.7670,
        0.3428,
        0.0000,
        -0.0746,
        0.7713,
        0.3706,
        -0.0128,
        -0.0796,
        1.3255,
        0.0752,
        0.0533,
        0.0203,
        0.0000,
        0.0496,
        0.1350,
        0.1303,
        0.0345,
        0.1277,
        0.1050,
        0.0392,
        0.0533,
        0.0786,
        -0.0056,
        0.0346,
        -0.0007,
        0.0683,
        -0.0082,
        0.0312,
    ]
)


# Motion Models
class CONST_ACC_MODEL:
    KF_DIM = [9, 6]

    # Measurement Matrix
    KF_H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )

    def STATE_VEC(init):
        return [init[0], init[1], init[2], init[3], init[4], init[5], 0, 0, 0]

    # State Transition Matrix
    def KF_F(dt):
        return np.array(
            [
                [1, 0, 0, dt, 0, 0, (0.5 * dt**2), 0, 0],
                [0, 1, 0, 0, dt, 0, 0, (0.5 * dt**2), 0],
                [0, 0, 1, 0, 0, dt, 0, 0, (0.5 * dt**2)],
                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    def KF_Q_DISCR(dt):
        return block_diag(
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
        )


class CONST_VEL_MODEL:
    KF_DIM = [6, 6]
    # Measurement Matrix
    KF_H = np.eye(6)

    def STATE_VEC(init):
        return [init[0], init[1], init[2], init[3], init[4], init[5]]

    # State Transition Matrix
    def KF_F(dt):
        return np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    def KF_Q_DISCR(dt):
        return block_diag(
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
        )


MOTION_MODEL = CONST_ACC_MODEL


JOINT_INDEX = {
    0: "SpineBase",
    1: "SpineMid",
    2: "Neck",
    3: "Head",
    4: "ShoulderLeft",
    5: "ElbowLeft",
    6: "WristLeft",
    7: "ShoulderRight",
    8: "ElbowRight",
    9: "WristRight",
    10: "HipLeft",
    11: "KneeLeft",
    12: "AnkleLeft",
    13: "FootLeft",
    14: "HipRight",
    15: "KneeRight",
    16: "AnkleRight",
    17: "FootRight",
    18: "SpineShoulder"
}

BODY_JOINTS = {
    "UPPER": [ 3, 2, 1, 18 ],
    "LOWER": [0, 10, 14, 15, 11], 
    "ARMS" : [4, 5, 6, 7, 8, 9]
}


## Experiment Constants (Code, Label)
EXP_LOC_CL = ("A", "Cluttered Room")
EXP_LOC_CR = ("B", "Common Room")
EXP_LOC_PL = ("C", "Penguin Lab")
EXP_LOC_SR = ("D", "Small Room")
EXP_LOCATIONS = [EXP_LOC_CL, EXP_LOC_CR, EXP_LOC_PL, EXP_LOC_SR]

EXP_OBJ_EMPTY = ("O", "No Object")
EXP_OBJ_MIR = ("A", "Mirror")
EXP_OBJ_PLY = ("B", "Plywood")
EXP_OBJ_MET = ("C", "Metal")
EXP_OBJECTS = [EXP_OBJ_EMPTY, EXP_OBJ_MIR, EXP_OBJ_PLY, EXP_OBJ_MET]


EXP_MOV_RAS = ("A","Right Arm Stretch")
EXP_MOV_LAS = ("B", "Left Arm  Stretch")
EXP_MOV_BAS = ("C", "Both Arms Waving")
EXP_MOV_BAUD = ("D", "Both Arms Up/Down")
EXP_MOV_SQT = ("E", "Squat")
EXP_MOV_BOW = ("F", "Bow")
EXP_MOVEMENTS = [EXP_MOV_RAS ,EXP_MOV_LAS ,EXP_MOV_BAS ,EXP_MOV_BAUD ,EXP_MOV_SQT ,EXP_MOV_BOW ]

EXP_POS_STR = ("A", "Straight")
EXP_POS_LFT = ("B", "Left")
EXP_POS_RGT = ("C", "Right")
EXP_POSITIONS = [EXP_POS_STR, EXP_POS_LFT, EXP_POS_RGT]

EXP_DIST_1 = ("A", "1m")
EXP_DIST_2 = ("B", "2m")
EXP_DIST_4 = ("C", "4m")
# EXP_DIST_75 = ("C", "7.5m")
EXP_DISTANCE = [EXP_DIST_1, EXP_DIST_2, EXP_DIST_4]


EXP_ANG_0 = ("A", "0 deg")
EXP_ANG_30R = ("B", "30 deg Right")
EXP_ANG_30L = ("C", "30 deg Left")
EXP_ANG_60R = ("D", "60 def Right")
EXP_ANG_60L = ("E", "60 def Left")

EXP_ANGLE = [EXP_ANG_0, EXP_ANG_30L, EXP_ANG_30R, EXP_ANG_60L, EXP_ANG_60R]


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



# Naming Exp: Loc + Mov + Pos + Dist + Ang + Obj, use O for no obj
import json
def get_exp_names():
    d = {}
    for loc in EXP_LOCATIONS:
        for mov in EXP_MOVEMENTS:
            for pos in EXP_POSITIONS:
                for dist in EXP_DISTANCE:
                    for ang in EXP_ANGLE:
                        for obj in EXP_OBJECTS:

                            if ang[0] == "A" and (obj[0] == "A" or obj[0] == "O"):
                                d[loc[0] + mov[0]+ pos[0] + dist[0] + ang[0]+obj[0]] = " ".join([loc[1], mov[1], pos[1], dist[1], ang[1],  obj[1]])
    with open("experiments.json", "w") as f:
        f.write(json.dumps(d))

# get_exp_names()