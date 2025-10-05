# HoloDepth: A VFX-Centric Pipeline for Multi-Plane Images (still in devlopment!)

HoloDepth is a complete, from-scratch implementation of a Multi-Plane Image (MPI) training and viewing pipeline, inspired by the 2019 Google DeepView paper. This project was born out of the frustration of trying to adapt brittle academic code for a real production workflow. It is designed from the ground up to be robust, debuggable, and practical for a VFX artist.

https://augmentedperception.github.io/deepview/

The pipeline allows you to train a neural network to understand the 3D structure of a scene from a set of images and camera poses, and then generate an interactive, web-based 3D viewer from the result.

## Core Features

- **From-Scratch Implementation:** The entire pipeline (`data_loader`, `model`, `utils`, `renderer`) is written from scratch for clarity and robustness.
- **Artist-Controlled Training:** Control key parameters like the number of MPI planes, the reference camera view, and validation "witness" cameras directly from the command line.
- **Manifest-Driven Pipeline:** Every training run generates a `training_manifest.json`, ensuring that the inference and viewing steps are perfectly consistent with the model's training conditions.
- **Efficient Hybrid Model:** Uses a custom MobileNet-UNet hybrid architecture, balancing training speed with high-quality output.
- **Standalone HTML Viewer:** Generates a single, self-contained `viewer.html` file with all data embedded, perfect for sharing and reviewing.

## The Workflow

The pipeline is divided into three main stages:

### 1. Training

Use `train.py` to train a model on your scene. You can specify the output directory, number of planes, and camera perspectives.

```bash
# Example: Train a 20-plane model using image 40 as the reference view.
python train.py ^
    --scenes_path "path/to/your/dataset" ^
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
    --scenes_path "path/to/your/dataset" ^
    --scene_name "scene_037"
```
This will create a folder at `mpi_output/scene_037` containing the transparent PNG slices and an `mpi_config.json` file.

### 3. Viewer Generation

Use `generate_viewer.py` to create the final, interactive HTML file.

```bash
python generate_viewer.py --mpi_dir "mpi_output/scene_037"
```
This will generate `viewer.html` in your project root. Double-click it to open the interactive viewer in your browser.
