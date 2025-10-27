#!/usr/bin/env python3
"""
run_sfm.py

Minimal PyCOLMAP pipeline to reconstruct a 3D scene from a folder of images.
Place this script in the root of your project (where 'images/' is located).
"""

from pathlib import Path
import pycolmap

# --- Paths ---
project_root = Path(__file__).parent
image_dir = project_root / "images"
database_path = project_root / "database.db"
output_path = project_root / "output"

# Create output folder if it doesn't exist
output_path.mkdir(exist_ok=True)

# --- Step 1: Extract features ---
print("🔹 Extracting features from images...")
pycolmap.extract_features(database_path, image_dir)

# --- Step 2: Match features ---
print("🔹 Matching features between images...")
pycolmap.match_exhaustive(database_path)

# --- Step 3: Incremental reconstruction (sparse 3D) ---
print("🔹 Reconstructing scene...")
reconstruction = pycolmap.incremental_mapping(database_path, image_dir, output_path)

# --- Step 4: Save reconstruction ---
reconstruction.write(output_path)
print(f"✅ Reconstruction saved to: {output_path}")
print("You can view it in COLMAP GUI or convert to PLY for Meshlab/Blender.")

