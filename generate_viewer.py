import os
import base64
import json
import argparse
import numpy as np
from PIL import Image

# This is the placeholder text we will look for in the HTML template.
# It should exist inside a <script> tag.
INJECT_PLACEHOLDER = '// <!-- INJECT_MPI_DATA_HERE -->'

def create_viewer_html(config):
    """
    Generates a standalone HTML viewer for a set of MPI slices.
    """
    print("--- Starting HoloDepth Viewer Generator ---")

    # --- 1. Find and Verify MPI Slices ---
    print(f"--- Searching for MPI slices in '{config['mpi_dir']}'... ---")
    if not os.path.isdir(config['mpi_dir']):
        print(f"!!! FATAL ERROR: MPI directory not found at '{config['mpi_dir']}'. Aborting. !!!")
        return

    slice_files = sorted([f for f in os.listdir(config['mpi_dir']) if f.endswith('.png')])
    if not slice_files:
        print(f"!!! FATAL ERROR: No .png slices found in '{config['mpi_dir']}'. Aborting. !!!")
        return
        
    print(f"--- Found {len(slice_files)} MPI slices. ---")

    # --- 2. Encode Slice Images to Base64 ---
    print("--- Encoding images to Base64... ---")
    base64_uris = []
    for filename in slice_files:
        filepath = os.path.join(config['mpi_dir'], filename)
        with open(filepath, 'rb') as f:
            binary_data = f.read()
            # The Base64 string is encoded to ASCII for embedding in the HTML.
            base64_string = base64.b64encode(binary_data).decode('ascii')
            # This is the standard format for an embedded image source.
            uri = f"data:image/png;base64,{base64_string}"
            base64_uris.append(uri)
    
    print("--- Image encoding complete. ---")

    # --- 3. Gather Blueprint Data (Depths & Intrinsics) ---
    print("--- Calculating depth planes and camera intrinsics... ---")
    
    # Calculate depths spaced linearly in disparity (1/depth), which is geometrically correct.
    # We use the same near/far planes as training.
    near_depth, far_depth = config['near_plane'], config['far_plane']
    num_planes = len(slice_files)
    disparity_space = np.linspace(1.0 / near_depth, 1.0 / far_depth, num_planes)
    plane_depths = (1.0 / disparity_space).tolist()

    # Load the original camera data to get the intrinsics of our reference camera (view 0).
    try:
        json_path = os.path.join(config['scene_dir'], 'models.json')
        with open(json_path, 'r') as f:
            camera_data = json.load(f)[0]
            ref_camera = camera_data[0] # The reference camera is always the first one
    except Exception as e:
        print(f"!!! FATAL ERROR: Could not read or parse '{json_path}'. Aborting. Error: {e} !!!")
        return

    # The intrinsics in models.json are for the original image resolution.
    # We MUST scale them to match the resolution of our MPI slices.
    mpi_image_path = os.path.join(config['mpi_dir'], slice_files[0])
    with Image.open(mpi_image_path) as img:
        mpi_w, mpi_h = img.size

    orig_w, orig_h = ref_camera['width'], ref_camera['height']
    orig_f, (orig_px, orig_py) = ref_camera['focal_length'], ref_camera['principal_point']

    scale_x = mpi_w / orig_w
    scale_y = mpi_h / orig_h

    # We assume focal length is in pixels and scales with width.
    focal_scaled = orig_f * scale_x
    px_scaled = orig_px * scale_x
    py_scaled = orig_py * scale_y

    camera_intrinsics = [focal_scaled, px_scaled, py_scaled]
    print("--- Blueprint data successfully gathered. ---")

    # --- 4. Assemble the JavaScript Payload ---
    # This dictionary contains all the "blueprint" data the viewer needs.
    mpi_payload = {
        "image_uris": base64_uris,
        "plane_depths": plane_depths,
        "camera_intrinsics": camera_intrinsics,
        "img_width": mpi_w,
        "img_height": mpi_h,
    }
    
    # Convert the Python dictionary to a JSON string for injection.
    # Using 'indent=2' makes the final HTML file more readable for debugging.
    js_payload_string = f"const mpi_config = {json.dumps(mpi_payload, indent=2)};"

    # --- 5. Inject Data into HTML Template ---
    print(f"--- Reading HTML template from '{config['template_path']}'... ---")
    try:
        with open(config['template_path'], 'r') as f:
            template_html = f.read()
    except Exception as e:
        print(f"!!! FATAL ERROR: Could not read template file. Aborting. Error: {e} !!!")
        return

    if INJECT_PLACEHOLDER not in template_html:
        print(f"!!! FATAL ERROR: Placeholder '{INJECT_PLACEHOLDER}' not found in template. Aborting. !!!")
        return

    print("--- Injecting data into template... ---")
    final_html = template_html.replace(INJECT_PLACEHOLDER, js_payload_string)

    # --- 6. Save the Final Viewer ---
    try:
        with open(config['output_path'], 'w') as f:
            f.write(final_html)
        print(f"--- Successfully generated viewer: '{config['output_path']}' ---")
        print("\n--- Mission Accomplished! ---")
    except Exception as e:
        print(f"!!! FATAL ERROR: Could not save final HTML file. Aborting. Error: {e} !!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a standalone HTML viewer for HoloDepth MPI slices.")
    
    # The folder containing the 'slice_xxx.png' files.
    parser.add_argument('--mpi_dir', type=str, required=True, help="Directory containing the generated MPI slices.")
    
    # The original scene folder, needed to read 'models.json' for camera data.
    parser.add_argument('--scene_dir', type=str, required=True, help="Path to the original scene directory (e.g., '.../scene_037').")
    
    # The name of the template file. Assumes it's in the project root.
    parser.add_argument('--template_path', type=str, default='deepview-mpi-viewer-template.html', help="Path to the HTML template file.")
    
    # The name of the final output file.
    parser.add_argument('--output_path', type=str, default='viewer.html', help="Path for the output HTML viewer file.")

    args = parser.parse_args()

    # Hard-coded near/far planes from the training configuration.
    config = {
        'mpi_dir': args.mpi_dir,
        'scene_dir': args.scene_dir,
        'template_path': args.template_path,
        'output_path': args.output_path,
        'near_plane': 1.0,
        'far_plane': 100.0,
    }

    create_viewer_html(config)