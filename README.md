# HoloDepth: A VFX-Centric Pipeline for Multi-Plane Images (still in development!)

HoloDepth is a complete, from-scratch implementation of a Multi-Plane Image (MPI) training and viewing pipeline, inspired by the 2019 Google paper on [DeepView: View Synthesis with Learned Gradient Descent](https://openaccess.thecvf.com/content_CVPR_2019/papers/Flynn_DeepView_View_Synthesis_With_Learned_Gradient_Descent_CVPR_2019_paper.pdf). This project was born out of the frustration of trying to adapt brittle academic code for a real production workflow. It is designed from the ground up to be robust, debuggable, and practical for a VFX artist.

The pipeline allows you to train a neural network to understand the 3D structure of a scene from a set of images and camera poses, and then generate an interactive, web-based 3D viewer from the result.

## Core Features

- **From-Scratch Implementation:** The entire pipeline (`data_loader`, `model`, `utils`, `renderer`) is written from scratch for clarity and robustness.
- **Artist-Controlled Training:** Control key parameters like the number of MPI planes, the reference camera view, and validation "witness" cameras directly from the command line.
- **Manifest-Driven Pipeline:** Every training run generates a `training_manifest.json`, ensuring that the inference and viewing steps are perfectly consistent with the model's training conditions.
- **Efficient Hybrid Model:** Uses a custom MobileNet-UNet hybrid architecture, balancing training speed with high-quality output.
- **Standalone HTML Viewer:** Generates a single, self-contained `viewer.html` file with all data embedded, perfect for sharing and reviewing.

## Installation

This project is built with Python 3.9 and PyTorch. The recommended way to set up the environment is with Conda, as it will handle all CUDA dependencies automatically.

### Option 1: Conda (Recommended)

1.  Ensure you have Miniconda or Anaconda installed.
2.  From the project's root directory, create and activate the environment using the provided file:

    ```bash
    # Create the environment
    conda env create -f environment.yml

    # Activate the environment
    conda activate holodepth
    ```

### Option 2: Pip

This method is for users who prefer `pip` and `venv`.

1.  Create and activate a new virtual environment (e.g., `python -m venv venv`).
2.  **Install PyTorch Manually:** This is a critical step. For GPU support, you must install PyTorch by following the official instructions for your specific OS and CUDA version at [pytorch.org](https://pytorch.org/).
3.  Once PyTorch is installed, install the remaining packages using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## The Workflow

The pipeline is divided into four main stages:

### 0. Data Preparation (Using PFTrack)

The pipeline requires a `models.json` file containing camera data for each scene. This project includes a `pftrack_to_holodepth.py` script to generate this file from a solved camera in PFTrack.

1.  Open the `pftrack_to_holodepth.py` script.
2.  Set the `OUTPUT_DIRECTORY` variable to the full path of your scene folder (which must contain an image sequence in a `cam` subfolder).
3.  Run the script from within PFTrack. It will generate the required `models.json` file in your scene directory.

### 1. Training

Use `train.py` to train a model on your scene. You can specify the output directory, number of planes, and camera perspectives.

```bash
# Example: Train a 20-plane model using image 40 as the reference view.
# On Windows, use ^ for line breaks. On Mac/Linux, use \.
python train.py ^
    --scenes_path "path/to/your/dataset/folder" ^
    --output_dir "training_output_20_planes" ^
    --num_planes 20 ^
    --ref_view 40 ^
    --witness_views 20 70 120
```
This will create the `training_output_20_planes` directory containing your trained model (`hybrid_mpi_model.pt`) and the crucial `training_manifest.json`.

### 2. MPI Generation (Inference)

Use `generate_mpi.py` to create the PNG slices from your trained model. This script is driven by the manifest file from the previous step.

```bash
python generate_mpi.py ^
    --training_dir "training_output_20_planes" ^
    --scenes_path "path/to/your/dataset/folder" ^
    --scene_name "scene_037"
```
This will create a folder at `mpi_output/scene_037` containing the transparent PNG slices and an `mpi_config.json` file.

### 3. Viewer Generation

Use `generate_viewer.py` to create the final, interactive HTML file.

```bash
python generate_viewer.py --mpi_dir "mpi_output/scene_037"
```
This will generate `viewer.html` in your project root. Double-click it to open the interactive viewer in your browser.
