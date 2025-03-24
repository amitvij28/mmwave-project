import sys
import numpy as np
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QPushButton
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import constants as const
from mpl_toolkits.mplot3d import Axes3D

# ---------- Set Experiment Here-----------
EXPERIMENT = "AEABAA"
JSON_PATH = f"{const.P_DATA_PATH}/sim/{EXPERIMENT}.json"

ERROR_PLOT_RANGE = 10
# -----------------------------------------

class MatplotlibGridWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Upper - 2,3, 18, 4, 7
         # Define connections and keypoints
        self.connections = [
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

        # Define keypoint colors
        self.keypoint_colors = [
            "blue",  # SpineBase,
            "blue",  # SpineMid,
            "blue",  # Neck,
            "red",  # Head,
            "blue",  # ShoulderLeft,
            "green",  # ElbowLeft,
            "green",  # WristLeft,
            "blue",  # ShoulderRight,
            "green",  # ElbowRight,
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


        # Create figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create layout and add canvas
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Initialize subplot grid
        self.axes = []
        self.create_grid()

        # Initial plot
        # self.update_plot(0)

    def setup_subplot(self, subplot: Axes3D):
        axis_dim = const.V_3D_AXIS
        subplot.set_xlim(axis_dim[0][0], axis_dim[0][1])
        subplot.set_ylim(axis_dim[1][0], axis_dim[1][1])
        subplot.set_zlim(axis_dim[2][0], axis_dim[2][1])
        subplot.set_xlabel("X")
        subplot.set_ylabel("Y")
        subplot.set_zlabel("Z")
        subplot.invert_yaxis()
        subplot.invert_xaxis()
        return subplot.scatter([], [], [])

    def create_grid(self):
        """Create a 3x3 grid of subplots."""
        self.figure.clear()
        grid_spec = plt.GridSpec(3,3, height_ratios=[2,2,1])

        self.ax_gt = self.figure.add_subplot(grid_spec[1], projection="3d")
        self.setup_subplot(self.ax_gt)
        self.axes.append(self.ax_gt)
        
        self.ax_ast = self.figure.add_subplot(grid_spec[3], projection="3d")
        self.setup_subplot(self.ax_ast)
        self.axes.append(self.ax_ast)
        
        self.ax_mars = self.figure.add_subplot(grid_spec[4], projection="3d")
        self.setup_subplot(self.ax_mars)
        self.axes.append(self.ax_mars)

        self.ax_ast_error = self.figure.add_subplot(grid_spec[6])
        self.ax_mars_error = self.figure.add_subplot(grid_spec[7])
        self.axes.extend([self.ax_ast_error, self.ax_mars_error])

        self.ax_gt.set_title("Ground Truth")
        self.ax_ast.set_title("Asterios Model")
        self.ax_mars.set_title("MARS Model")
        self.ax_ast_error.set_title("Asterios Error")
        self.ax_mars_error.set_title("MARS Error")


    def update_skeleton(self, reshaped_data, ax, multiple_tracks=False):
        tracks = [reshaped_data] if not multiple_tracks else reshaped_data
        for track in tracks:
            for connection in self.connections:
                keypoint_1 = connection[0]
                keypoint_2 = connection[1]

                x_values = [float(track[0][keypoint_1]), float(track[0][keypoint_2])]
                y_values = [float(track[2][keypoint_1]), float(track[2][keypoint_2])]
                z_values = [float(track[1][keypoint_1]), float(track[1][keypoint_2])]
                ax.plot(x_values, y_values, z_values, color="black")

            for keypoint_index in range(len(track[0])):
                color = self.keypoint_colors[keypoint_index]
                marker = (
                    "o" if keypoint_index != 3 else "s"
                )  # Use square marker for the head
                ax.scatter(
                    float(track[0][keypoint_index]),
                    float(track[2][keypoint_index]),
                    float(track[1][keypoint_index]),
                    c=color,
                    marker=marker,
                    s=15 if keypoint_index == 3 else 15,  # Larger size for the head
                )

    def plot_error_graph(self, errors, index, ax, key):
        x_values = [i for i in range(ERROR_PLOT_RANGE)]
        if index > 5:
            x_values = [i for i in range(index-5, index+5)]

        selected_val = x_values.index(index)

        mean_x_error = [ errors[i][key]['x'] for i in range(ERROR_PLOT_RANGE)]
        mean_y_error = [ errors[i][key]['y'] for i in range(ERROR_PLOT_RANGE)]
        mean_z_error = [ errors[i][key]['z'] for i in range(ERROR_PLOT_RANGE)]

        ax.plot(x_values, mean_x_error, label = "Mean X Error")
        ax.plot(x_values, mean_y_error, label = "Mean Y Error")
        ax.plot(x_values, mean_z_error, label = "Mean Z Error")
        ax.axvline(x=index, color='r', linestyle='--')
        ax.set_ylim(0, 1.0)
        # ax.plot()
        ax.legend()


    def update_plot(self, gt, ast, mars, errors, index):
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        self.setup_subplot(self.ax_gt)
        self.setup_subplot(self.ax_mars)
        self.setup_subplot(self.ax_ast)

        self.ax_gt.set_title("Ground Truth")
        self.ax_ast.set_title("Asterios Model")
        self.ax_mars.set_title("MARS Model")
        self.ax_ast_error.set_title("Asterios Error")
        self.ax_mars_error.set_title("MARS Error")

        # Different graphs in each subplot
        self.update_skeleton(gt, self.ax_gt)
        self.update_skeleton(mars, self.ax_mars)
        self.update_skeleton(ast, self.ax_ast, True)


        if errors is not None:
            self.plot_error_graph(errors, index, self.ax_ast_error, 'ast')
            self.plot_error_graph(errors, index, self.ax_mars_error, 'mars')



        # for i, ax in enumerate(self.axes):
        #     ax.legend()
        
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self, slider_length, experiment_data):
        super().__init__()

        self.main_data = experiment_data
        self.setWindowTitle("mmWave Pose Estimation Analysis")
        self.setGeometry(100, 100, 900, 800)

        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create grid plot widget
        self.matplotlib_widget = MatplotlibGridWidget()

        # Create slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(slider_length)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.update_graphs)

        # Create play and pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_slider)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_slider)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_slider)



        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.matplotlib_widget)
        layout.addWidget(self.slider)
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        self.central_widget.setLayout(layout)

        self.update_graphs(0)

    def update_graphs(self, index):
        
        if index < 5:
            errors = self.main_data['error'][:10]
        else:
            errors = self.main_data['error'][index-5:index+5]
        
        errors = None
        self.matplotlib_widget.update_plot(gt=self.main_data['gt'][index],ast= self.main_data['ast'][index], mars=self.main_data['mars'][index], errors=errors, index=index)

    
    def play_slider(self):
        self.timer.start(100)  # Adjust the interval to change speed

    def pause_slider(self):
        self.timer.stop()

    def advance_slider(self):
        current_value = self.slider.value()
        if current_value < self.slider.maximum():
            self.slider.setValue(current_value + 1)
        else:
            self.timer.stop() 


def read_experiment_data():
    data = None
    with open(JSON_PATH, "r") as f:
        data = json.loads(f.read())
    return data


if __name__ == "__main__":

    experiment_data = read_experiment_data()
    slider_length = len(experiment_data['gt'])
    print(slider_length)
    app = QApplication(sys.argv)
    window = MainWindow(slider_length, experiment_data)
    window.show()
    sys.exit(app.exec_())
