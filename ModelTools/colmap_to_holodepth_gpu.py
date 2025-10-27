#!/usr/bin/env python3
"""
colmap_to_holodepth.py (Definitive Version - Final API Correction)

Workflow:
  1. Activate the tool's environment: `conda activate colmap_tools_env`
  2. Run the script from the project root: `python ModelTools/colmap_to_holodepth.py`
"""
import pycolmap
from pathlib import Path
import json
import numpy as np
import sys

# --- Configuration ---
DATASET_BASE_DIR_SUFFIX = "dataset/data/newset"
# ---------------------


def qvec_to_rotmat(quat):
    """
    Manual, from-scratch conversion of a quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,   2*x*z + 2*y*w],
        [2*x*y + 2*z*w,   1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,   2*y*z + 2*x*w,   1 - 2*x*x - 2*y*y]
    ])


def process_single_scene(scene_path: Path):
    print(f"\n{'='*60}\n▶️  Processing Scene: {scene_path.name}\n{'='*60}")
    
    image_dir = scene_path / "cam"
    database_path = scene_path / "colmap.db"
    output_sfm_dir = scene_path / "colmap_sparse"
    output_json_path = scene_path / "models.json"

    if not (image_dir.is_dir() and any(image_dir.iterdir())):
        print(f"❌ ERROR: 'cam' subfolder not found or is empty in {scene_path}. Skipping.")
        return False

    output_sfm_dir.mkdir(exist_ok=True)
    
    use_gpu = pycolmap.has_cuda
    if use_gpu: print("✅ GPU acceleration enabled.")
    else: print("ℹ️ Running on CPU.")

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    pycolmap.extract_features(str(database_path), str(image_dir), device=device)
    pycolmap.match_exhaustive(str(database_path), device=device)

    print("🔹 Starting COLMAP scene reconstruction...")
    reconstructions = pycolmap.incremental_mapping(str(database_path), str(image_dir), str(output_sfm_dir))

    if not reconstructions:
        print("❌ FATAL: COLMAP reconstruction failed.")
        return False
    
    best_rec_id = max(reconstructions, key=lambda k: len(reconstructions[k].images))
    reconstruction = reconstructions[best_rec_id]

    print(f"✅ COLMAP reconstruction successful with {len(reconstruction.images)} registered images.")
    return convert_colmap_to_holodepth_json(reconstruction, output_json_path)


def convert_colmap_to_holodepth_json(reconstruction, output_json_path: Path):
    camera_data_list = []
    sorted_image_ids = sorted(reconstruction.images.keys())
    cam_id = list(reconstruction.cameras.keys())[0]
    camera = reconstruction.cameras[cam_id]
    
    pp = [camera.principal_point_x, camera.principal_point_y]
    focal_length = camera.focal_length
    width, height = camera.width, camera.height

    print(f"🔹 Using camera intrinsics: Focal={focal_length:.2f}, PP=({pp[0]:.2f}, {pp[1]:.2f})")

    for image_id in sorted_image_ids:
        image = reconstruction.images[image_id]
        
        # THE FINAL, GUARANTEED FIX:
        # 1. Call the method to get the pose object.
        pose = image.cam_from_world()
        
        # 2. Get the raw quaternion from the pose object's .rotation attribute.
        #    The raw data is in an array called .quat on the rotation object.
        quat = pose.rotation.quat
        
        # 3. Use our manual math function to create the rotation matrix.
        R = qvec_to_rotmat(quat)

        # 4. Get the translation vector from the pose object.
        t = pose.translation
        
        # 5. Calculate the camera's world position.
        position = -R.T @ t
        # END OF FIX

        angle = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))

        if np.isclose(angle, 0):
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2 * np.sin(angle))
        
        view_data = {
            "relative_path": f"cam/{image.name}", "width": width, "height": height,
            "focal_length": focal_length, "pixel_aspect_ratio": 1.0,
            "principal_point": [float(p) for p in pp],
            "position": position.tolist(),
            "orientation": (axis * angle).tolist()
        }
        camera_data_list.append(view_data)
        
    final_json_structure = [camera_data_list]
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_structure, f, indent=4)
        
    print(f"✅ Successfully wrote models.json with {len(camera_data_list)} camera views.")
    return True


def run_interactive_scanner(scan_path: Path):
    print(f"🔎 Scanning for scenes in: {scan_path}...")
    valid_scenes = sorted([p for p in scan_path.iterdir() if p.is_dir() and (p / "cam").is_dir()], key=lambda p: p.name)

    if not valid_scenes:
        print("❌ No valid scenes found.")
        return

    while True:
        print("\n✨ Found the following potential scenes:")
        for i, scene_path in enumerate(valid_scenes):
            print(f"  [{i+1}] {scene_path.name}")
        
        print("\nWhich scene(s) would you like to process?")
        print("➡️  Enter a number (e.g., 1), numbers separated by commas (e.g., 1,3), 'all', or 'q' to quit.")
        choice = input("> ").strip().lower()

        if choice == 'q':
            print("👋 Exiting.")
            break
        
        scenes_to_process = []
        try:
            if choice == 'all':
                scenes_to_process = valid_scenes
            else:
                indices = [int(i.strip()) - 1 for i in choice.split(',')]
                if any(i < 0 or i >= len(valid_scenes) for i in indices):
                    print("⚠️ Invalid number detected.")
                    continue
                scenes_to_process = [valid_scenes[i] for i in indices]
        except ValueError:
            print("⚠️ Invalid input.")
            continue
        
        success_count = 0
        for scene in scenes_to_process:
            if process_single_scene(scene):
                success_count += 1
        
        print(f"\n🎉 --- Batch Complete ---")
        print(f"Successfully processed {success_count} of {len(scenes_to_process)} selected scene(s).")
        break


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent 
    scan_path = project_root / DATASET_BASE_DIR_SUFFIX
    
    if not scan_path.is_dir():
        print(f"❌ FATAL ERROR: The dataset directory was not found at {scan_path}")
        sys.exit(1)
        
    run_interactive_scanner(scan_path)


if __name__ == '__main__':
    main()