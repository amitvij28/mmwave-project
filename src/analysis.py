import json
import os
import numpy as np
from sklearn import metrics
import math
import constants as const
import matplotlib.pyplot as plt

# ##########--------------Constants--------------#########
# TODO: Run this for all the experiments
# EXPERIMENT = "B1"
SIMULATION_DIR = f"{const.P_DATA_PATH}/sim"
NO_OF_HUMANS = 1
START_TIME = 0.25
END_TIME = 0.75
# --------------------------------------------------------

def get_experiment_type(experiment):
    # print(f"experiment: {experiment}")
    experiment_split = experiment
    # print(experiment_split)
    location = [ loc for loc in const.EXP_LOCATIONS if loc[0] == experiment_split[0]][0]
    movement = [ mov for mov in const.EXP_MOVEMENTS if mov[0] == experiment_split[1]][0]
    position = [pos for pos in const.EXP_POSITIONS if pos[0] == experiment_split[2]][0]
    distance = [dist for dist in const.EXP_DISTANCE if dist[0] == experiment_split[3]][0]
    angle = [ang for ang in const.EXP_ANGLE if ang[0] == experiment_split[4]][0]
    objects = [obj for obj in const.EXP_OBJECTS if obj[0] == experiment_split[5]][0]
    return {"loc": location, "mov":movement, "pos":position, "dist":distance, "ang":angle, "obj": objects}

def plot_line_graph(data, xlabel, ylabel, legend, title, path):
    plt.figure(figsize=(8, 5))  # Set figure size
    plt.plot(range(len(data)), data, marker='o', linestyle='-', color='b', markersize=6, label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)  # Show grid for better visibility
    plt.savefig(path)
    plt.legend()
    plt.close()

def plot_cdf_grid(data, model_name):
    movements = list(data.keys())
    distances = list(data[movements[0]].keys())
    locations = list(data[movements[0]][distances[0]].keys())

    fig, axes = plt.subplots(len(distances), len(movements), figsize=(len(movements)*2, len(distances)*2))
    fig.suptitle(f"CDF Plots for {model_name}", fontsize=16)

    for i, movement in enumerate(movements):
        fig.text(0.12 + i * (0.76 / len(movements)), 1.01, movement, ha="center", fontsize=12, fontweight="bold")

    for j, distance in enumerate(distances):
        fig.text(-0.02, 0.85 - j * (0.76 / len(distances)), distance, va="center", ha="center", 
                 fontsize=12, fontweight="bold", rotation=90, transform=fig.transFigure)


    for i in range(len(movements)):
        for j in range(len(distances)):
            ax = axes[j, i]
            d1 = data[movements[i]][distances[j]][locations[0]]
            d2 = data[movements[i]][distances[j]][locations[1]]
            # Filter to not go beyond 1m
            d1 = list(filter( lambda x: x < 0.5 , d1))
            d2 = list(filter( lambda x: x < 0.5, d2))
            d1_sorted = np.sort(d1)
            d2_sorted = np.sort(d2)
            cdf1 = np.linspace(0, 1, len(d1_sorted))
            cdf2 = np.linspace(0, 1, len(d2_sorted))

            # Plot CDFs
            l1 = [l[1] for l in const.EXP_LOCATIONS if l[0] == locations[0]][0]
            l2 = [l[1] for l in const.EXP_LOCATIONS if l[0] == locations[1]][0]
            dist = [d[1] for d in const.EXP_DISTANCE if d[0] == distances[j]][0]
            move = [m[1] for m in const.EXP_MOVEMENTS if m[0] == movements[i]][0]
            ax.plot(d1_sorted, cdf1, label=f"{l1}")
            ax.plot(d2_sorted, cdf2, label=f"{l2}", linestyle="dashed")
            ax.set_xlim([0, 0.5])
            # ax.set_title(f"{movements[i]}, {distances[i]}", fontsize=8)
            ax.set_xlabel("Error (m)")
            ax.set_ylabel("Cumulative Probability")
            if i == 0:
                ax.set_ylabel(dist, fontsize=10, fontweight="bold")  # Row labels
            if j == len(distances) - 1:
                ax.set_xlabel(move, fontsize=10, fontweight="bold")  # Column labels

            ax.legend(fontsize=6)
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.savefig(f"./plots/cdf/{model_name}")

# def plot_cdf(ax, d1, d2, d1_label, d2_label, experiment):
#     sorted_ast = np.sort(d1)
#     sorted_mars = np.sort(d2)
#     cdf = np.linspace(0, 1, len(sorted_ast))
#     # plt.figure(figsize=(8, 5))
#     ax.plot(sorted_ast, cdf,  color='red', label=d1_label)
#     ax.plot(sorted_mars, cdf, color='blue', label=d2_label)
#     ax.xlabel("Error (m)")
#     ax.ylabel("Cumulative Probability")
#     ax.title("Cumulative Distribution Function (CDF)")
#     ax.grid()
#     ax.legend()
#     # plt.savefig(f"./plots/{experiment}")
#     # plt.close()

def translate_skeleton(ground_truth, prediction):
    translation_vector = ground_truth[:, 0] - prediction[:, 0]
    translated_pred = prediction + translation_vector[:, np.newaxis]
    return translated_pred


def calc_mean_error(ground_truth, prediction):
    translated_prediction = translate_skeleton(ground_truth, prediction)
    error_per_joints = {}
    joint_errors = []
    for joint_id in list(const.JOINT_INDEX.keys()):
        gt_x = ground_truth[0, joint_id]
        gt_y = ground_truth[2, joint_id]
        gt_z = ground_truth[1, joint_id]

        pred_x = translated_prediction[0, joint_id]
        pred_y = translated_prediction[2, joint_id]
        pred_z = translated_prediction[1, joint_id]

        err = metrics.mean_absolute_error([gt_x, gt_y, gt_z], [pred_x, pred_y, pred_z])
        error_per_joints[joint_id] = err
        joint_errors.append(err)

    mean_error = np.mean(joint_errors)

    return {"error_per_joints": error_per_joints, "mean_error": mean_error}


def read_experiment(experiment):
    data = None
    with open(f"{SIMULATION_DIR}/{experiment}", "r") as f:
        data = json.loads(f.read())

    return data


def process_data(data):
    sim_len = len(data['gt'])
    gt_data = []
    ast_data = []
    mars_data = []
    for idx in range(sim_len):
        gt_data.append( np.array(data['gt'][idx]))
        mars_data.append(np.array(data['mars'][idx]))
        tracks = []
        for track in data['ast'][idx]:
            tracks.append(np.array(track))
        ast_data.append(tracks)
    return gt_data, ast_data, mars_data


def error_calculation(gt_data, ast_data, mars_data):
    # ASTERIOS MODEL: Choosing a correct track (for multiple reflections)
    track_centroids = []
    chosen_track = None
    for track in ast_data:
        track_centroids.append(np.mean(track, axis=1))    
    min_index = 0
    if len(ast_data) == NO_OF_HUMANS:
        chosen_track = ast_data[0]
    elif len(ast_data) == 0:
        chosen_track = None
    else:
        gt_centroid = np.mean(gt_data, axis=1)
        min_diff = math.inf
        for index in range(len(track_centroids)):
            diff = np.linalg.norm(track_centroids[index] - gt_centroid)
            if diff < min_diff:
                min_diff = diff
                min_index = index
        chosen_track = ast_data[min_index]
    ast_error = None
    ast_jt_err = None
    if chosen_track is not None:
        ast_error = calc_mean_error(gt_data, chosen_track)
        ast_jt_err = joint_analysis(ast_error['error_per_joints'])
    
    mars_error = calc_mean_error(gt_data, mars_data)
    mars_jt_err = joint_analysis(mars_error['error_per_joints'])

    return {"ast":ast_error, "mars": mars_error, "ast_track_len": len(ast_data), "ast_jt": ast_jt_err, "mars_jt": mars_jt_err}

# Find which part of the body is causing the most issues
def joint_analysis(error):
    joint_error = {}
    for body_area in list(const.BODY_JOINTS.keys()):
        jts = const.BODY_JOINTS[body_area]
        for j in jts:
            if body_area not in joint_error:
                joint_error[body_area] = []
            joint_error[body_area].append(error[j])
        
    for body_area in list(joint_error.keys()):
        joint_error[body_area] = np.mean(joint_error[body_area])
    return joint_error

def calc_jt_mean(data):
    mean_errors = {}
    num_experiments = len(data)
    for key in data[0]: 
        mean_errors[key] = sum(exp[key] for exp in data) / num_experiments
    return mean_errors
    

def run_analysis(data, experiment):
    print(f"Running analysis for {experiment}")
    gt_data, ast_data, mars_data = process_data(data)
    error_data = []
    ast_mean_err = []
    mars_mean_err = []
    ast_track_len = []
    ast_jt_err = []
    mars_jt_err = []
    # Mean absolute joint error calculation
    for idx in range(len(gt_data[int(len(gt_data)*START_TIME): int(len(gt_data)*END_TIME)])):
        error = error_calculation(gt_data[idx], ast_data[idx], mars_data[idx])        
        error_data.append(error)
        if error['ast'] is not None:
            ast_mean_err.append(error['ast']['mean_error'])
            mars_mean_err.append(error['mars']['mean_error'])
            ast_jt_err.append(error['ast_jt'])
            mars_jt_err.append(error['mars_jt'])
        ast_track_len.append(len(ast_data[idx]))
       
    # print(len(error_data))

    # plot_cdf(ast_mean_err, mars_mean_err, experiment)
    # plot_line_graph(ast_track_len, "time","Human count" ,"No. of Humans Predicted", "Asterios Model: Reflection Analysis", f"./plots/tracks/{experiment}")
    # print(ast_jt_err)
    # print(mars_jt_err)
    return {"ast_mean_err": ast_mean_err, "mars_mean_err": mars_mean_err, "ast_track_len": ast_track_len, "ast_jt": calc_jt_mean(ast_jt_err), "mars_jt": calc_jt_mean(mars_jt_err)}
    # print(ast_mean_err)
    

# def plot_error_mean(data1, data2, labels=('Dataset 1', 'Dataset 2')):
#     """
#     Plots the mean error with error bars (standard deviation) for two datasets.
    
#     Parameters:
#     - data1: list or numpy array of numerical values (first dataset)
#     - data2: list or numpy array of numerical values (second dataset)
#     - labels: tuple of labels for the datasets (default: ('Dataset 1', 'Dataset 2'))
#     """
#     means = [np.mean(data1), np.mean(data2)]
#     std_devs = [np.std(data1, ddof=1), np.std(data2, ddof=1)]  # ddof=1 for sample std deviation
    
#     x_labels = [labels[0], labels[1]]
#     x_pos = np.arange(len(x_labels))
    
#     plt.figure(figsize=(6, 4))
#     plt.bar(x_pos, means, yerr=std_devs, capsize=5, alpha=0.7, color=['blue', 'green'])
#     plt.xticks(x_pos, x_labels)
#     plt.ylabel('Mean Value')
#     plt.title('Mean Error Plot with Standard Deviation')
#     plt.grid(axis='y', linestyle='--', alpha=0.6)
#     plt.savefig("./plots/cdf/mean_model")


def plot_error_mean(data1, data2, data3, data4, labels=('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4')):
    """
    Plots a boxplot for four datasets.
    
    Parameters:
    - data1: list or numpy array of numerical values (first dataset)
    - data2: list or numpy array of numerical values (second dataset)
    - data3: list or numpy array of numerical values (third dataset)
    - data4: list or numpy array of numerical values (fourth dataset)
    - labels: tuple of labels for the datasets (default: ('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'))
    """
    data = [data1, data2, data3, data4]
    
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.ylabel('Values')
    plt.title('Boxplot of Four Datasets')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("./plots/cdf/mean_model")

def store_reflections(experiments):
    exp_list = {}
    with open("experiments.json", "r") as f:
        exp_list = json.loads(f.read())
    
    refl_list = {}
    for exp in experiments:
        refl_list[exp] = exp_list[exp]
    
    with open("./plots/cdf/reflections.json", "w") as f:
        f.write(json.dumps(refl_list))

def plot_single_jt(ax, experiment_results):
    # Extract error values for each body area
    body_areas = list(experiment_results[0].keys())  # ["UPPER", "LOWER", "ARMS"]
    data = [[exp[area] for exp in experiment_results] for area in body_areas]

    # Create box plot
    ax.boxplot(data, labels=body_areas, patch_artist=True, boxprops=dict(facecolor="lightblue"))

    # Labels
    ax.set_xlabel("Body Area")
    ax.set_ylabel("Error")


def plot_joint_errors(error, is_env=True):
    rows = ["ast", "mars"]
    cols = list(error.keys())
    col_names = []
    
    check_arr = const.EXP_LOCATIONS if is_env else const.EXP_MOVEMENTS
    for c in cols:
        col_names.append([e[1] for e in check_arr if e[0] == c][0])

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(len(cols) * 3, len(rows) * 3))
    
    # Add row and column headings
    for j, col_name in enumerate(col_names):
        axes[0, j].set_title(col_name, fontsize=10, fontweight="bold")  # Column headings

    for i, row_name in enumerate(rows):
        axes[i, 0].annotate(row_name.upper(), xy=(-0.8, 0.5), xycoords='axes fraction',
                            fontsize=10, fontweight="bold", ha="right", va="center",
                            rotation=90)  # Row headings

    # Plot data
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax = axes[i, j]  
            r = rows[i]
            c = cols[j]
            experiment_results = error[c][r]
            plot_single_jt(ax, experiment_results)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.savefig(f"./plots/cdf/joint_{'env' if is_env else 'mov'}")

def run():
    experiment_data = {}
    experiments = os.listdir(SIMULATION_DIR)
    for experiment in experiments:
        # if experiment[0] != "A":
        #     continue
        data = read_experiment(experiment)
        if data is None:
            print(f"No data found for {experiment}!")
            return
        experiment_name = experiment.split(".")[0]
        experiment_analysis = run_analysis(data, experiment_name)
        experiment_data[experiment_name] = experiment_analysis
    

    ast_cdf_data = {}
    mars_cdf_data = {}
    ast_errs = []
    mars_errs = []
    exp_tracks = []
    cl_ast_errs = []
    pl_ast_errs = []
    cl_mars_errs = []
    pl_mars_errs = []
    
    jt_env_error = {}
    jt_mov_error = {}

    for experiment in list(experiment_data.keys()):
        exp_data = experiment_data[experiment]
        exp_type = get_experiment_type(experiment)
        
        for track_len in exp_data['ast_track_len']:
            if track_len > NO_OF_HUMANS:
                exp_tracks.append(experiment)
                break
        
        if exp_type["loc"][0] not in jt_env_error:
            jt_env_error[exp_type['loc'][0]] = {"ast":[], "mars":[]}
        jt_env_error[exp_type['loc'][0]]['ast'].append(exp_data['ast_jt'])
        jt_env_error[exp_type['loc'][0]]['mars'].append(exp_data['mars_jt'])


        if exp_type['mov'][0] not in jt_mov_error:
            jt_mov_error[exp_type['mov'][0]] = {"ast": [], "mars": []}
        jt_mov_error[exp_type['mov'][0]]['ast'].append(exp_data['ast_jt'])
        jt_mov_error[exp_type['mov'][0]]['mars'].append(exp_data['mars_jt'])

        if exp_type["mov"][0] not in ast_cdf_data:
            ast_cdf_data[exp_type["mov"][0]] = {}
            mars_cdf_data[exp_type["mov"][0]] = {}
        

        if exp_type['dist'][0] not in ast_cdf_data[exp_type["mov"][0]]:
            ast_cdf_data[exp_type["mov"][0]][exp_type['dist'][0]] = {}
        if exp_type['dist'][0] not in mars_cdf_data[exp_type["mov"][0]]:
            mars_cdf_data[exp_type["mov"][0]][exp_type['dist'][0]] = {}
        a =  ast_cdf_data[exp_type["mov"][0]][exp_type['dist'][0]]
        m = mars_cdf_data[exp_type["mov"][0]][exp_type['dist'][0]]
        

        if exp_type['loc'][0] not in a:
            a[exp_type['loc'][0]] = []
        if exp_type['loc'][0] not in m:
            m[exp_type['loc'][0]] = []
        
        a[exp_type['loc'][0]].extend(exp_data['ast_mean_err'])
        m[exp_type["loc"][0]].extend(exp_data['mars_mean_err'])

        ast_errs.extend(exp_data['ast_mean_err'])
        mars_errs.extend(exp_data['mars_mean_err'])

        if exp_type['loc'][0] == "A":
            cl_ast_errs.extend(exp_data['ast_mean_err'])
            cl_mars_errs.extend(exp_data['mars_mean_err'])
        elif exp_type['loc'][0] == "C":
            pl_ast_errs.extend(exp_data['ast_mean_err'])
            pl_mars_errs.extend(exp_data['mars_mean_err'])
    
    # CDF Curve for Asterios
    plot_cdf_grid(ast_cdf_data, "Asterios")
    # CDF Curve for MARS
    plot_cdf_grid(mars_cdf_data, "MARS")

    # Err b/w Asterios and MARS
    plot_error_mean(cl_ast_errs, cl_mars_errs, pl_ast_errs, pl_mars_errs, ("Cluttered Room\nAsterios", "Cluttered Room\nMARS ", "Penguin Lab\n Asterios", "Penguin Lab\nMARS"))

    # Err b/w Environments

    # Joint Err % b/w Environments & Models
    plot_joint_errors(jt_env_error)
    # Joint Err b/w Movements & Models
    plot_joint_errors(jt_mov_error, False)
    # Reflections Found
    store_reflections(exp_tracks)

      


run()
    

# TODO:
#   e. Change the code to be classes and extendable