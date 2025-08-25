# An Empirical Study on the Generalization of mmWave-Based Human Pose Estimation

This repository contains the complete codebase, dataset, and analysis scripts for the Master of Science thesis, "An Empirical Study on the Generalization of mmWave-Based Human Pose Estimation," completed at Delft University of Technology.

## Abstract

Millimeter-wave (mmWave) radar is a promising technology for Human Pose Estimation (HPE), offering privacy and robustness in challenging environmental conditions. However, its reliability is often hindered by poor generalization to scenarios, users, and movements unseen during training. This thesis presents a comprehensive empirical study to analyze and quantify the dimensions causing this poor generalization. To enable this research, we introduce **`mmDiverse`**, a new, large-scale, systematically structured dataset. Using this dataset, we evaluate two foundational models (Baseline MARS and Temporal MARS) through a series of targeted experiments designed to isolate the impact of four key dimensions: **environment**, **movement**, **user**, and **distance**. Our findings reveal that user diversity is the most critical challenge, and we provide a set of evidence-based guidelines for developing more resilient mmWave-based HPE systems.

## Key Contributions

1.  **A Novel Dataset (`mmDiverse`):** A large-scale, publicly available mmWave dataset specifically designed for generalization research, containing a rich variety of motions, users, and environments.
2.  **Systematic Generalization Analysis:** A rigorous, empirical study that isolates and quantifies the effects of real-world dimensions on the performance of mmWave HPE models.
3.  **Evidence-Based Guidelines:** A set of actionable guidelines for researchers and developers to inform future data collection strategies and model architecture choices for more robust HPE systems.
4.  **Reproducible Codebase:** The complete code for data gathering, pre-processing, model training, and analysis is provided to ensure full reproducibility of this study.

## The `mmDiverse` Dataset

The `mmDiverse` dataset is the core of this research. It was collected using a Texas Instruments IWR1443 mmWave sensor and a Microsoft Kinect V2 for ground-truth skeletal data. The dataset is structured along four key dimensions:

* **Environment:** 3 distinct indoor environments with varying levels of clutter and multi-path interference (Clean, Medium Clutter, High Clutter).
* **Movement:** 6 distinct upper-body movements designed to test compositional, symmetrical, and kinematic complexity (e.g., Right Arm Raise, Left Arm Wave, Combined Arms Wave).
* **User:** 4 participants with varying heights to analyze the impact of inter-user body shape and kinematic variations.
* **Distance:** 2 distinct distances from the sensor (2 meters and 3 meters) to evaluate robustness to signal attenuation and data sparsity.

The full, pre-processed dataset is available for download at [LINK TO DATASET - e.g., Zenodo, Google Drive].

## Repository Structure

This repository is organized as follows:

```
.
├── data_gathering/         # Scripts for capturing data from mmWave and Kinect sensors
├── preprocessing/          # Scripts for data synchronization, cleaning, and formatting
├── models/                 # Implementations of the Baseline MARS and Temporal MARS models
├── experiments/            # Scripts to run the nine generalization experiments
├── analysis/               # Jupyter notebooks for generating plots and insights
├── amitvij_6005217_msc_cs_thesis.pdf # The full thesis document
└── README.md               # This file
```

## Setup and Usage

### Prerequisites

* Python 3.8+
* TensorFlow 2.x
* Pandas, NumPy, Scikit-learn, Matplotlib

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/amitvij28/mmwave-project.git](https://github.com/amitvij28/mmwave-project.git)
    cd mmwave-project
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Experiments

1.  **Download the Dataset:** Download the `mmDiverse` dataset from [LINK TO DATASET] and place it in a `data/` directory.

2.  **Pre-process the Data:** (Optional, if using raw data) Run the pre-processing scripts to format the data for the models.
    ```bash
    python preprocessing/main_preprocessor.py
    ```

3.  **Run an Experiment:** Navigate to the `experiments` directory and run one of the experiment scripts. For example, to run the "Lab-to-World (E3)" experiment:
    ```bash
    python experiments/run_experiment_E3.py
    ```

