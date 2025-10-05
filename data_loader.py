import os
import json
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

def load_scene_data(scenes_base_path, scene_name, resolution_w, resolution_h):
    """
    Loads all relevant data for a single scene from the Spaces dataset.
    This is the corrected version that properly handles relative paths from models.json.
    """
    scene_path = os.path.join(scenes_base_path, scene_name)
    json_path = os.path.join(scene_path, 'models.json')

    try:
        with open(json_path, 'r') as f:
            camera_data = json.load(f)[0]
    except IOError as e:
        print(f"Error: Could not open {json_path}. {e}")
        return [] # Return empty list if json not found

    transform = transforms.Compose([
        transforms.Resize((resolution_h, resolution_w), Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    all_views_data = []

    for i, view in enumerate(camera_data):
        # ======================================================================
        # --- THE FIX ---
        # 1. Get the relative path directly from the JSON file.
        relative_path_from_json = view['relative_path']

        # 2. Construct the full, correct path by joining the scene path
        #    with the relative path. This is the simple, correct way.
        final_img_path = os.path.join(scene_path, relative_path_from_json)
        # ======================================================================

        if not os.path.exists(final_img_path):
            print(f"Warning: Image for view {i} at '{final_img_path}' not found. Skipping.")
            continue

        try:
            img_pil = Image.open(final_img_path).convert("RGB")
            img_tensor = transform(img_pil)
        except Exception as e:
            print(f"Warning: Could not load or transform image {final_img_path}. Error: {e}. Skipping.")
            continue


        position = torch.tensor(view['position'], dtype=torch.float32)
        orientation_axis_angle = torch.tensor(view['orientation'], dtype=torch.float32)

        angle = torch.linalg.norm(orientation_axis_angle)
        if angle > 1e-6:
            axis = orientation_axis_angle / angle
            skew = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=torch.float32)
            rotation_matrix = torch.eye(3) + torch.sin(angle) * skew + (1 - torch.cos(angle)) * torch.matmul(skew, skew)
        else:
            rotation_matrix = torch.eye(3, dtype=torch.float32)

        pose = torch.eye(4, dtype=torch.float32)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = position

        h, w = view['height'], view['width']
        focal = view['focal_length']
        px, py = view['principal_point']

        scale_x = resolution_w / w
        scale_y = resolution_h / h

        focal_x_scaled = focal * scale_x
        focal_y_scaled = focal * view['pixel_aspect_ratio'] * scale_y
        px_scaled = px * scale_x
        py_scaled = py * scale_y

        intrinsics = torch.tensor([[focal_x_scaled, 0, px_scaled], [0, focal_y_scaled, py_scaled], [0, 0, 1]], dtype=torch.float32)

        all_views_data.append({'image': img_tensor, 'pose': pose, 'intrinsics': intrinsics})

    return all_views_data