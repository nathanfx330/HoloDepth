# Project: HoloDepth - PFTrack Exporter
# Version: 5.0 (GitHub Release)
# Description: This script runs inside PFTrack to export camera data to the


import pfpy
import os
import sys

try:
    import json
except ImportError:
    import simplejson as json
import math

# ==============================================================================
# --- User Settings ---
# ==============================================================================

# 1. SET YOUR OUTPUT DIRECTORY:
#    This is the full path to your scene folder. The 'models.json' file will be
#    saved directly inside this directory.
#    Example (Windows): r"E:\HoloDepth\spaces_dataset\data\newset\scene_037"
#    Example (Mac/Linux): "/path/to/my/project/scene_037"
OUTPUT_DIRECTORY = r"C:\path\to\your\scene_folder"

# 2. IMAGE SEQUENCE SETTINGS:
#    These settings must match your image files.
IMAGE_SUBFOLDER = "cam"      # The folder inside OUTPUT_DIRECTORY containing your images.
SEQUENCE_PREFIX = "img_"     # The prefix of your image files (e.g., "img_").
SEQUENCE_PADDING = "####"    # The number of digits for the frame number (e.g., "####" for img_0001).
IMAGE_EXTENSION = "jpg"      # The file extension of your images (e.g., "jpg", "png").

# ==============================================================================
# --- Script ---
# ==============================================================================

# --- Setup ---
PADDING_WIDTH = SEQUENCE_PADDING.count("#")
OUTPUT_FILENAME = "models.json"

# --- Debug ---
def debug(msg):
    print str(msg)
    sys.stdout.flush()

# --- Math (Unchanged) ---
def matrix_multiply(A, B):
    C = [[0,0,0],[0,0,0],[0,0,0]];
    for i in range(3):
        for j in range(3):
            for k in range(3): C[i][j]+=A[i][k]*B[k][j]
    return C
def euler_to_rotation_matrix(rotation):
    r, p, y = [math.radians(a) for a in rotation]
    Rx=[[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]]
    Ry=[[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]]
    Rz=[[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]]
    return matrix_multiply(Rx, matrix_multiply(Ry, Rz))
def rotation_matrix_to_axis_angle(R):
    trace=R[0][0]+R[1][1]+R[2][2]; clipped=max(-1.0,min(1.0,(trace-1.0)/2.0))
    angle=math.acos(clipped)
    if abs(angle)<1e-6: return [0.0,0.0,0.0]
    rx,ry,rz=R[2][1]-R[1][2],R[0][2]-R[2][0],R[1][0]-R[0][1]; axis=[rx,ry,rz]
    norm=math.sqrt(sum(c*c for c in axis));
    if norm<1e-6: norm=1.0
    axis=[c/norm for c in axis]; return [c*angle for c in axis]

# --- Main Export Logic ---
def export_for_holodepth():
    debug("--- HoloDepth PFTrack Exporter v5.0 ---")

    # 1. Validate User Settings
    if "C:\\path\\to" in OUTPUT_DIRECTORY:
        debug("ERROR: Please set the 'OUTPUT_DIRECTORY' variable at the top of the script.")
        return
    if not os.path.isdir(OUTPUT_DIRECTORY):
        debug("ERROR: The specified OUTPUT_DIRECTORY does not exist: %s" % OUTPUT_DIRECTORY)
        return
    
    image_folder_path = os.path.join(OUTPUT_DIRECTORY, IMAGE_SUBFOLDER)
    if not os.path.isdir(image_folder_path):
        debug("ERROR: The image subfolder '%s' was not found inside your OUTPUT_DIRECTORY." % IMAGE_SUBFOLDER)
        return

    # 2. Get Camera Data from PFTrack
    try:
        cam = pfpy.getCameraRef(0)
    except:
        debug("ERROR: Could not get a valid PFTrack camera.")
        return

    start_frame, end_frame = cam.getInPoint(), cam.getOutPoint()
    width, height = cam.getFrameWidth(), cam.getFrameHeight()
    debug("Found camera tracking data for frames %d - %d." % (start_frame, end_frame))

    # 3. Process Each Frame
    camera_data_list = []
    for frame in xrange(start_frame, end_frame + 1):
        frame_index = frame - start_frame + 1

        # A. Construct image filenames and paths
        image_filename = "%s%0*d.%s" % (SEQUENCE_PREFIX, PADDING_WIDTH, frame_index, IMAGE_EXTENSION)
        
        # B. Create the clean, RELATIVE path for the JSON file (e.g., "cam/img_0001.jpg")
        relative_path_for_json = (IMAGE_SUBFOLDER + "/" + image_filename).replace("\\", "/")

        # C. Create the full, absolute path for verification purposes only
        absolute_path_for_checking = os.path.join(image_folder_path, image_filename)

        if not os.path.exists(absolute_path_for_checking):
            debug("WARNING: Image file not found for frame %d: %s" % (frame, absolute_path_for_checking))

        # D. Extract camera parameters
        try:
            rotation = cam.getEulerRotation(frame, 'xyz')
            orientation = rotation_matrix_to_axis_angle(euler_to_rotation_matrix(rotation))
            
            camera_info = {
                "relative_path": relative_path_for_json,
                "width": float(width),
                "height": float(height),
                "principal_point": [width / 2.0, height / 2.0],
                "focal_length": cam.getFocalLength(frame, 'pixels'),
                "pixel_aspect_ratio": 1.0,
                "position": cam.getTranslation(frame),
                "orientation": orientation,
            }
            camera_data_list.append(camera_info)
        except Exception, e:
            debug("WARNING: Could not get camera data for frame %d. Error: %s" % (frame, str(e)))

    if not camera_data_list:
        debug("ERROR: No camera data was exported. Aborting.")
        return

    # 4. Write the Final JSON File
    final_output = [camera_data_list]
    output_filepath = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)

    try:
        with open(output_filepath, "w") as f:
            f.write(json.dumps(final_output, indent=4))
        debug("SUCCESS: Wrote %d frames to %s" % (len(camera_data_list), output_filepath))
    except Exception, e:
        debug("ERROR: Could not write the output file. Error: %s" % str(e))

# --- Run Script ---
if __name__ == '__main__':
    export_for_holodepth()
