import argparse
import os
import sys
import cv2 # Requires opencv-python
import numpy as np
from tqdm import tqdm # Requires tqdm
import glob # Added for finding .sens files
import concurrent.futures # Added for parallel processing

from src.metadata_generation.ScanNet.preprocess.SensorData import SensorData

# Assuming SensorData is in the same directory or PYTHONPATH is set
# If SensorData.py is elsewhere, adjust the import accordingly
# try:
#     # Assuming SensorData.py is in the same directory as the script using it or in the path
#     # Adjust the path if SensorData is located differently, e.g., from ..SensorData import SensorData
#     from .SensorData import SensorData
# except ImportError as e:
#     print(f"Error: Could not import SensorData class: {e}")
#     print("Make sure SensorData.py is accessible (e.g., in the same directory or added to PYTHONPATH).")
#     # Attempt import from parent directory if running from preprocess/scannet
#     try:
#         # This assumes the script is run from its directory and SensorData.py is one level up
#         sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#         from SensorData import SensorData
#         print("Successfully imported SensorData from parent directory.")
#     except ImportError:
#          # If SensorData.py is in the preprocess/scannet directory itself
#          try:
#              # This assumes the script and SensorData.py are in preprocess/scannet
#              # No change needed if SensorData.py is in the same dir and PYTHONPATH is set correctly
#              # Check current dir import again, maybe structure is flatter
#              from SensorData import SensorData # Re-try import
#              print("Successfully imported SensorData from current directory.")
#          except ImportError:
#              print("Error: SensorData class not found even after checking parent/current directories.")
#              sys.exit(1)


# params
parser = argparse.ArgumentParser(description="Export color images from ScanNet .sens files to video.")
# # data paths
# # parser.add_argument('--filename', required=True, help='path to .sens file') # Removed
# parser.add_argument('--scans_dir', required=True, help='path to the directory containing scene subdirectories (e.g., data/scannet/scans)')
# # parser.add_argument('--output_video', required=True, help='path to output video file (e.g., output/video.mp4)') # Removed
# parser.add_argument('--output_dir', required=True, help='path to the directory where output videos will be saved')
# parser.add_argument('--train_val_splits_path', required=False, help='path to the directory containing train/val split files (e.g., scannetv2_train.txt, scannetv2_val.txt)') # Added
# # video settings
# parser.add_argument('--width', type=int, required=True, help='output video width')
# parser.add_argument('--height', type=int, required=True, help='output video height')
# parser.add_argument('--fps', type=int, default=30, help='output video frame rate (default: 30)')
# parser.add_argument('--frame_skip', type=int, default=1, help='process every nth frame (default: 1)')
# parser.add_argument('--codec', type=str, default='mp4v', help='video codec (default: mp4v for .mp4). Use "avc1" for H.264.')
# parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of processes to use for parallel processing (default: number of cores)')

def export_scene_video(sens_file_path, output_video_path, width, height, fps, frame_skip, codec):
    """Exports video for a single scene's .sens file."""
    print(f"Processing scene: {os.path.basename(os.path.dirname(sens_file_path))}")
    print(f'Loading {sens_file_path}...')
    try:
        sd = SensorData(sens_file_path)
    except NameError:
         print("Error: SensorData class failed to be imported.")
         return False # Indicate failure
    except Exception as e:
        print(f"Error loading SensorData from {sens_file_path}: {e}")
        return False # Indicate failure
    print(f'Loaded {len(sd.frames)} frames.')

    if not hasattr(sd, 'frames') or not sd.frames:
        print("Error: SensorData object does not contain frames or is empty.")
        return False

    # Check compression type - important for calling the right decompressor
    if not hasattr(sd, 'color_compression_type'):
         print("Error: SensorData object does not have 'color_compression_type' attribute.")
         print("Please check the SensorData class definition.")
         return False

    # Check if frames have the required decompress method
    if not hasattr(sd.frames[0], 'decompress_color'):
         print("Error: Frame objects in SensorData do not have the 'decompress_color' method.")
         print("Please check the SensorData class definition.")
         return False


    image_size = (width, height) # (width, height) for OpenCV
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, image_size)

    if not writer.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}.")
        print("Check if the codec is supported and paths are correct.")
        return False

    print(f"Exporting video to {output_video_path} with resolution {image_size} and FPS {fps}...")

    frame_indices = range(0, len(sd.frames), frame_skip)
    processed_frames = 0
    skipped_frames = 0
    success = True # Track if processing completes without fatal errors

    try:
        for i in tqdm(frame_indices, desc="Processing frames", leave=False): # leave=False for nested loops
            try:
                frame = sd.frames[i]
                # Decompress color image using the type specified in SensorData header
                color_image_rgb = frame.decompress_color(sd.color_compression_type)

            except IndexError:
                 print(f"Warning: Frame index {i} out of bounds. Stopping scene processing.")
                 success = False # Consider this a failure for the scene
                 break
            except Exception as e:
                print(f"Warning: Could not decompress color image for frame index {i}: {e}. Skipping frame.")
                skipped_frames += 1
                continue

            if color_image_rgb is None:
                print(f"Warning: Decompressed null color image for frame index {i}. Skipping frame.")
                skipped_frames += 1
                continue

            # Ensure image is NumPy array and convert RGB to BGR for OpenCV
            if not isinstance(color_image_rgb, np.ndarray):
                 try:
                     color_image_rgb = np.array(color_image_rgb)
                 except Exception as e_conv:
                     print(f"Warning: Could not convert decompressed image to NumPy array for frame {i}: {e_conv}. Skipping frame.")
                     skipped_frames += 1
                     continue

            if len(color_image_rgb.shape) == 3 and color_image_rgb.shape[2] == 3:
                 color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
            elif len(color_image_rgb.shape) == 2: # Grayscale
                 color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_GRAY2BGR)
            elif len(color_image_rgb.shape) == 3 and color_image_rgb.shape[2] == 4:
                 color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGBA2BGR)
            else:
                 print(f"Warning: Unexpected image format/shape ({color_image_rgb.shape}) for frame {i}. Skipping frame.")
                 skipped_frames += 1
                 continue

            # Resize image if necessary
            current_height, current_width = color_image_bgr.shape[:2]
            if current_width != width or current_height != height:
                 resized_image = cv2.resize(color_image_bgr, image_size, interpolation=cv2.INTER_AREA)
            else:
                 resized_image = color_image_bgr

            # Write frame
            writer.write(resized_image)
            processed_frames += 1

    except AttributeError as e:
         print(f"Error: Problem accessing frame data or methods for {sens_file_path}: {e}")
         print("Please check the SensorData class structure.")
         success = False # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred during processing for {sens_file_path}: {e}")
        success = False # Indicate failure
    finally:
        if writer.isOpened():
            writer.release()

    if success:
        print(f'Finished exporting video for scene.')
        print(f'Processed {processed_frames} frames.')
        if skipped_frames > 0:
            print(f'Skipped {skipped_frames} frames due to errors.')
        print(f'Video saved to: {output_video_path}')
    else:
        print(f"Failed to fully export video for scene {os.path.basename(os.path.dirname(sens_file_path))}.")
        # Clean up partially created video file if process failed significantly
        if os.path.exists(output_video_path) and processed_frames == 0: # Or some other condition
             print(f"Deleting potentially incomplete file: {output_video_path}")
             try:
                 os.remove(output_video_path)
             except OSError as e_del:
                 print(f"Error deleting file {output_video_path}: {e_del}")

    return success

def main():
    opt = parser.parse_args()
    print("Script Options:")
    print(opt)

    sens_file_path = "/data/Datasets/ScanNet/scans/scene0000_00/scene0000_00.sens"
    output_video_path = "/data/Datasets/ScanNet/scans/scene0000_00/video.mp4"
    
    opt.width = 640
    opt.height = 480
    opt.fps = 24
    opt.frame_skip = 1
    opt.codec = "mp4v"
    opt.max_workers = 32
    export_scene_video(sens_file_path, output_video_path, opt.width, opt.height, opt.fps, opt.frame_skip, opt.codec)


if __name__ == '__main__':
    main() 