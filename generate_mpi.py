import torch
import numpy as np
import os
import argparse
import json
from PIL import Image

import data_loader, utils, model

def generate_mpi_slices(config):
    print("--- Starting MPI Generation ---")
    
    manifest_path = os.path.join(config['training_dir'], 'training_manifest.json')
    print(f"--- Loading training manifest from '{manifest_path}'... ---")
    try:
        with open(manifest_path, 'r') as f: manifest = json.load(f)
    except Exception as e:
        print(f"!!! FATAL ERROR: Could not load manifest. Aborting. Error: {e} !!!"); return
        
    device = torch.device('cuda' if torch.cuda.is_available() and not config['cpu'] else 'cpu')
    print(f"--- RUNNING ON {device.type.upper()} ---")

    model_path = os.path.join(config['training_dir'], 'hybrid_mpi_model.pt')
    print(f"\n--- Loading trained model from '{model_path}'... ---")
    try:
        net = model.HybridMPIModel(num_views=len(manifest['source_view_indices']), num_planes=manifest['num_planes']).to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        print("--- Model loaded successfully. ---")
    except Exception as e:
        print(f"!!! FATAL ERROR: Could not load model. Aborting. Error: {e} !!!"); return

    print(f"\n--- Loading scene data for '{config['scene_name']}'... ---")
    try:
        full_scene_data = data_loader.load_scene_data(config['scenes_path'], config['scene_name'], manifest['resolution_w'], manifest['resolution_h'])
        source_views = [full_scene_data[i] for i in manifest['source_view_indices']]
        # --- MODIFIED: Use the reference view from the manifest ---
        ref_view = full_scene_data[manifest['ref_view_idx']]
        print("--- Scene data loaded successfully. ---")
    except Exception as e:
        print(f"!!! FATAL ERROR: Could not load scene data. Aborting. Error: {e} !!!"); return

    src_imgs = torch.stack([v['image'] for v in source_views]).unsqueeze(0).to(device)
    src_poses = torch.stack([v['pose'] for v in source_views]).unsqueeze(0).to(device)
    src_intrinsics = torch.stack([v['intrinsics'] for v in source_views]).unsqueeze(0).to(device)
    ref_pose = ref_view['pose'].unsqueeze(0).unsqueeze(0).to(device)
    ref_intrinsics = ref_view['intrinsics'].unsqueeze(0).unsqueeze(0).to(device)
    plane_depths = torch.tensor(np.linspace(manifest['near_plane'], manifest['far_plane'], manifest['num_planes']), dtype=torch.float32, device=device)

    print("\n--- Generating MPI from source views... ---")
    with torch.no_grad():
        psv = utils.create_psv(src_imgs, src_poses, src_intrinsics, ref_pose, ref_intrinsics, plane_depths)
        predicted_mpi_rgba = net(psv)

    output_dir = os.path.join("mpi_output", config['scene_name'])
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Saving {predicted_mpi_rgba.shape[1]} MPI slices to '{output_dir}'... ---")
    
    # [Rest of the file is unchanged, including saving mpi_config.json]
    mpi_slices = predicted_mpi_rgba.squeeze(0); slice_filenames = []
    for i in range(mpi_slices.shape[0]):
        slice_tensor = mpi_slices[i]; slice_tensor[:, :, :3] = (slice_tensor[:, :, :3] * 0.5 + 0.5).clamp(0, 1)
        slice_np = (slice_tensor.cpu().numpy() * 255).astype(np.uint8); img = Image.fromarray(slice_np, 'RGBA')
        filename = f"slice_{i:03d}.png"; img_path = os.path.join(output_dir, filename); img.save(img_path)
        slice_filenames.append(filename)
    print(f"--- Saving MPI configuration file... ---")
    disparity_space = np.linspace(1.0 / manifest['near_plane'], 1.0 / manifest['far_plane'], manifest['num_planes'])
    viewer_depths = (1.0 / disparity_space).tolist()
    ref_cam_full_res_data = data_loader.load_scene_data(config['scenes_path'], config['scene_name'], -1, -1, use_cache=False)[manifest['ref_view_idx']]
    mpi_config_data = { "slice_filenames": slice_filenames, "plane_depths": viewer_depths,
        "ref_camera_intrinsics_full_res": { "focal_length": ref_cam_full_res_data['intrinsics'][0,0].item(),
            "principal_point": [ref_cam_full_res_data['intrinsics'][0,2].item(), ref_cam_full_res_data['intrinsics'][1,2].item()],
            "width": int(ref_cam_full_res_data['width']), "height": int(ref_cam_full_res_data['height']) } }
    with open(os.path.join(output_dir, "mpi_config.json"), 'w') as f: json.dump(mpi_config_data, f, indent=4)
    print(f"\n--- MPI generation complete. ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate MPI slices using a trained model and its manifest.")
    parser.add_argument('--training_dir', type=str, required=True, help="Directory containing the trained model and manifest.json.")
    parser.add_argument('--scenes_path', type=str, required=True, help="Path to the root of the dataset (e.g., '.../newset').")
    parser.add_argument('--scene_name', type=str, required=True, help="Name of the scene to process (e.g., 'scene_037').")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    generate_mpi_slices(vars(args))