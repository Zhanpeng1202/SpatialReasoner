import argparse
import os
import sys
import cv2  # Requires opencv-python
import numpy as np
from tqdm import tqdm  # Requires tqdm
import glob
import concurrent.futures
import math
import png # Requires pypng
import zipfile # Added for zip file handling
import shutil # Added for file copying

from src.metadata_generation.ScanNet.preprocess.SensorData import SensorData

# Helper function to parse scene_id.txt
def parse_scene_meta_file(filename):
    """Parses the scene_id.txt file to extract axis alignment and depth intrinsics."""
    metadata = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(' = ')
                if len(parts) == 2:
                    key, value = parts
                    metadata[key.strip()] = value.strip()

        # Extract and reshape axis alignment matrix
        axis_align_str = metadata.get('axisAlignment')
        if axis_align_str:
            axis_align_vals = list(map(float, axis_align_str.split()))
            if len(axis_align_vals) == 16:
                axis_align_matrix = np.array(axis_align_vals).reshape(4, 4)
            else:
                print(f"Warning: Invalid number of values for axisAlignment in {filename}. Expected 16, got {len(axis_align_vals)}.")
                axis_align_matrix = None
        else:
            print(f"Warning: axisAlignment key not found in {filename}.")
            axis_align_matrix = None

        # Construct depth intrinsics matrix
        try:
            fx = float(metadata['fx_depth'])
            fy = float(metadata['fy_depth'])
            mx = float(metadata['mx_depth'])
            my = float(metadata['my_depth'])
            depth_intrinsics = np.array([
                [fx, 0, mx, 0],
                [0, fy, my, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        except KeyError as e:
            print(f"Warning: Missing depth intrinsic key {e} in {filename}.")
            depth_intrinsics = None
        except ValueError as e:
            print(f"Warning: Invalid value for depth intrinsic key in {filename}: {e}.")
            depth_intrinsics = None

        return axis_align_matrix, depth_intrinsics

    except FileNotFoundError:
        print(f"Warning: Scene metadata file not found: {filename}")
        return None, None
    except Exception as e:
        print(f"Warning: Error parsing scene metadata file {filename}: {e}")
        return None, None

def save_matrix_to_file(matrix, filename):
    """Saves a numpy matrix to a text file."""
    with open(filename, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

def export_scene_sampled_frames(sens_file_path, output_base_dir, num_frames_to_sample, split, image_size=None, min_valid_components_per_frame_initial=0, min_valid_frames_per_scene=1):
    """Exports uniformly sampled pose, depth, color, instance mask, and intrinsics for a single scene, applying axis alignment to poses."""
    scene_id = os.path.basename(os.path.dirname(sens_file_path))
    scene_dir_path = os.path.dirname(sens_file_path)
    print(f"Processing scene: {scene_id} (split: {split})")
    print(f'Loading {sens_file_path}...')
    try:
        sd = SensorData(sens_file_path)
    except NameError:
        print("Error: SensorData class failed to be imported.")
        return None 
    except Exception as e:
        print(f"Error loading SensorData from {sens_file_path}: {e}")
        return None 
    print(f'Loaded {len(sd.frames)} frames.')

    if not hasattr(sd, 'frames') or not sd.frames:
        print("Error: SensorData object does not contain frames or is empty.")
        return None
    
    total_raw_frames = len(sd.frames)
    if total_raw_frames == 0:
        print(f"Scene {scene_id} has 0 frames. Skipping.")
        return scene_id 

    # --- Locate additional input files ---
    scene_meta_path = os.path.join(scene_dir_path, f"{scene_id}.txt")
    instance_zip_path = os.path.join(scene_dir_path, f"{scene_id}_2d-instance-filt.zip")
    if not os.path.exists(instance_zip_path):
         instance_zip_path = os.path.join(scene_dir_path, f"{scene_id}_2d-instance.zip")

    instance_zip = None
    if os.path.exists(instance_zip_path):
        try:
            instance_zip = zipfile.ZipFile(instance_zip_path, 'r')
            print(f"Opened instance mask zip: {instance_zip_path}")
        except zipfile.BadZipFile:
            print(f"Warning: Bad zip file: {instance_zip_path}. Instance masks will be unavailable.")
            instance_zip = None
        except Exception as e:
            print(f"Warning: Error opening zip file {instance_zip_path}: {e}. Instance masks will be unavailable.")
            instance_zip = None
    else:
        print(f"Warning: Instance mask zip file not found at {instance_zip_path} or fallback. Instance masks will be unavailable.")

    # --- Phase 1: Validate all available frames to create a list of candidates ---
    print(f"Validating all {total_raw_frames} available frames for scene {scene_id}...")
    valid_frame_candidates_info = [] 
    
    for original_idx in tqdm(range(total_raw_frames), desc=f"Validating raw frames for {scene_id}", leave=False):
        try:
            frame = sd.frames[original_idx]
            num_available_components = 0

            if hasattr(frame, 'camera_to_world') and frame.camera_to_world is not None:
                num_available_components += 1
            
            if hasattr(frame, 'decompress_depth') and hasattr(sd, 'depth_compression_type') and sd.depth_compression_type.lower() != 'unknown':
                num_available_components += 1

            if hasattr(frame, 'decompress_color') and hasattr(sd, 'color_compression_type') and sd.color_compression_type.lower() == 'jpeg':
                num_available_components += 1
            
            instance_mask_potentially_available = False
            if instance_zip:
                mask_filename_in_zip = f'instance-filt/{original_idx}.png' 
                try:
                    instance_zip.getinfo(mask_filename_in_zip) 
                    instance_mask_potentially_available = True
                except KeyError:
                    instance_mask_potentially_available = False 
                except Exception: 
                    instance_mask_potentially_available = False
            
            if instance_mask_potentially_available:
                num_available_components +=1

            if num_available_components >= min_valid_components_per_frame_initial:
                 valid_frame_candidates_info.append({"original_idx": original_idx})

        except IndexError:
            print(f"Warning: Raw frame index {original_idx} out of bounds during validation for scene {scene_id}.")
            continue 
        except Exception as e_val:
            print(f"Warning: Error validating raw frame {original_idx} for scene {scene_id}: {e_val}. Skipping candidate.")
            continue
            
    if not valid_frame_candidates_info:
        print(f"No valid frame candidates found for scene {scene_id} after initial validation. Skipping scene processing.")
        if instance_zip: instance_zip.close()
        return None 

    num_candidates = len(valid_frame_candidates_info)
    print(f"Found {num_candidates} valid frame candidates for scene {scene_id}.")

    # --- Sampling Logic based on validated candidates ---
    if num_candidates <= num_frames_to_sample:
        selected_candidate_indices = np.arange(num_candidates)
    else:
        selected_candidate_indices = np.linspace(0, num_candidates - 1, num_frames_to_sample, dtype=int)
        selected_candidate_indices = np.unique(selected_candidate_indices)

    actual_indices_to_process = [valid_frame_candidates_info[i]["original_idx"] for i in selected_candidate_indices]
    if not actual_indices_to_process: # Should not happen if valid_frame_candidates_info is not empty
        print(f"No frames selected for processing for scene {scene_id} after sampling (num_candidates: {num_candidates}, num_frames_to_sample: {num_frames_to_sample}). Skipping.")
        if instance_zip: instance_zip.close()
        return None
    print(f"Selected {len(actual_indices_to_process)} frames for export. First few original indices: {actual_indices_to_process[:10]}...")


    # --- Create output directories ---
    pose_output_dir = os.path.join(output_base_dir, 'pose', split, scene_id)
    depth_output_dir = os.path.join(output_base_dir, 'depth', split, scene_id)
    color_output_dir = os.path.join(output_base_dir, 'color', split, scene_id)
    instance_output_dir = os.path.join(output_base_dir, 'instance', split, scene_id)
    intrinsic_base_output_dir = os.path.join(output_base_dir, 'intrinsic', split)
    
    os.makedirs(pose_output_dir, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(color_output_dir, exist_ok=True)
    os.makedirs(instance_output_dir, exist_ok=True)
    os.makedirs(intrinsic_base_output_dir, exist_ok=True)

    # --- Process scene-level data ---
    axis_align_matrix, intrinsics_matrix = parse_scene_meta_file(scene_meta_path)
    if intrinsics_matrix is not None:
        intrinsic_out_filename = os.path.join(intrinsic_base_output_dir, f'intrinsic_depth_{scene_id}.txt')
        save_matrix_to_file(intrinsics_matrix, intrinsic_out_filename)
        print(f"Saved intrinsics to {intrinsic_out_filename}")
    else:
        print(f"Warning: Could not read/parse or save intrinsics for scene {scene_id} from {scene_meta_path}.")

    if axis_align_matrix is None:
        print(f"Warning: Could not read/parse axis alignment matrix for scene {scene_id} from {scene_meta_path}. Poses will NOT be aligned.")

    # --- Phase 2: Process and Save Selected Frames ---
    processed_attempt_count = 0 
    final_valid_frames_count = 0 
    total_skipped_components_in_export = 0 
    min_successful_components_per_frame_export = min_valid_components_per_frame_initial
    
    try:
        for original_idx in tqdm(actual_indices_to_process, desc=f"Exporting selected frames for {scene_id}", leave=False):
            try:
                frame = sd.frames[original_idx] 
                components_skipped_this_frame_export = 0
                
                pose_saved_path = None
                depth_saved_path = None
                color_saved_path = None
                instance_saved_path = None
                color_export_failed_critically = False

                # Export Pose
                if hasattr(frame, 'camera_to_world') and frame.camera_to_world is not None:
                    pose_matrix_original = frame.camera_to_world
                    pose_filename = os.path.join(pose_output_dir, f'{original_idx:06d}.txt')
                    if axis_align_matrix is not None:
                        try:
                            aligned_pose = np.dot(axis_align_matrix, pose_matrix_original)
                            save_matrix_to_file(aligned_pose, pose_filename)
                            pose_saved_path = pose_filename
                        except ValueError as e:
                            print(f"Warning: Error applying axis alignment for frame {original_idx}: {e}. Saving original.")
                            save_matrix_to_file(pose_matrix_original, pose_filename)
                            pose_saved_path = pose_filename 
                        except Exception as e:
                            print(f"Warning: Unexpected error during pose alignment/saving for frame {original_idx}: {e}. Skip pose comp.")
                            components_skipped_this_frame_export += 1
                    else:
                        save_matrix_to_file(pose_matrix_original, pose_filename)
                        pose_saved_path = pose_filename
                else:
                    print(f"Warning: Pose data not found for pre-validated frame {original_idx} during export. Skip pose comp.")
                    components_skipped_this_frame_export += 1

                # Export Depth
                if hasattr(frame, 'decompress_depth') and hasattr(sd, 'depth_compression_type') and sd.depth_compression_type.lower() != 'unknown':
                    depth_data = frame.decompress_depth(sd.depth_compression_type)
                    if depth_data is not None:
                        try:
                            depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(sd.depth_height, sd.depth_width)
                            depth_filename = os.path.join(depth_output_dir, f'{original_idx:06d}.png')
                            depth_image_reshaped = depth_image.reshape(-1, depth_image.shape[1]).tolist()
                            with open(depth_filename, 'wb') as f_png:
                                writer = png.Writer(width=depth_image.shape[1], height=depth_image.shape[0], bitdepth=16, greyscale=True)
                                writer.write(f_png, depth_image_reshaped)
                            depth_saved_path = depth_filename
                        except (ValueError, AttributeError, Exception) as e: # Catch specific and general errors
                            print(f"Warning: Error processing/writing depth for frame {original_idx}: {e}. Skip depth comp.")
                            components_skipped_this_frame_export += 1
                    else:
                        print(f"Warning: Decompressed null depth data for frame {original_idx}. Skip depth comp.")
                        components_skipped_this_frame_export += 1
                else:
                    print(f"Warning: Depth data/decompression not available for pre-validated frame {original_idx}. Skip depth comp.")
                    components_skipped_this_frame_export += 1

                # Export Color
                if hasattr(frame, 'decompress_color') and hasattr(sd, 'color_compression_type') and sd.color_compression_type.lower() == 'jpeg':
                    color_image = frame.decompress_color(sd.color_compression_type)
                    if color_image is not None:
                        try:
                            color_filename = os.path.join(color_output_dir, f'{original_idx:06d}.jpg')
                            if image_size:
                                color_image_resized = cv2.resize(color_image, (image_size[1], image_size[0]))
                            else:
                                color_image_resized = color_image
                            color_image_bgr = cv2.cvtColor(color_image_resized, cv2.COLOR_RGB2BGR)
                            if not cv2.imwrite(color_filename, color_image_bgr):
                                raise ValueError(f"cv2.imwrite failed for color {color_filename}")
                            color_saved_path = color_filename
                        except Exception as e:
                            print(f"Warning: Error saving color image for frame {original_idx}: {e}. CRITICAL: Skip color & FRAME.")
                            components_skipped_this_frame_export += 1
                            color_export_failed_critically = True
                    else:
                        print(f"Warning: Decompressed null color data for frame {original_idx}. CRITICAL: Skip color & FRAME.")
                        components_skipped_this_frame_export += 1
                        color_export_failed_critically = True
                else:
                    print(f"Warning: Color data/decompression not JPEG for pre-validated frame {original_idx}. CRITICAL: Skip color & FRAME.")
                    components_skipped_this_frame_export += 1
                    color_export_failed_critically = True

                if color_export_failed_critically:
                    if pose_saved_path and os.path.exists(pose_saved_path): os.remove(pose_saved_path)
                    if depth_saved_path and os.path.exists(depth_saved_path): os.remove(depth_saved_path)
                    total_skipped_components_in_export += (4 - components_skipped_this_frame_export) # Add remaining potential skips
                    processed_attempt_count += 1
                    continue 

                # Export Instance Mask
                if instance_zip:
                    mask_filename_in_zip = f'instance-filt/{original_idx}.png'
                    instance_output_filename = os.path.join(instance_output_dir, f'{original_idx:06d}.png')
                    try:
                        with instance_zip.open(mask_filename_in_zip, 'r') as mask_file:
                            mask_data = mask_file.read()
                        instance_mask = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_UNCHANGED)
                        if instance_mask is None: raise ValueError(f"cv2.imdecode failed for {mask_filename_in_zip}")
                        if image_size:
                            target_h, target_w = image_size
                            instance_mask = cv2.resize(instance_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        if not cv2.imwrite(instance_output_filename, instance_mask):
                            raise ValueError(f"cv2.imwrite failed for instance {instance_output_filename}")
                        instance_saved_path = instance_output_filename
                    except KeyError:
                        print(f"Warning: Instance mask {mask_filename_in_zip} not in zip for frame {original_idx}. Skip instance comp.")
                        components_skipped_this_frame_export += 1
                    except Exception as e:
                        print(f"Warning: Error processing instance mask {mask_filename_in_zip} for frame {original_idx}: {e}. Skip instance comp.")
                        components_skipped_this_frame_export += 1
                elif not instance_zip: # Instance masks were not available for the scene
                    pass # Not a skip for this frame if unavailable for scene, handled by num_successfully_exported

                processed_attempt_count += 1
                total_skipped_components_in_export += components_skipped_this_frame_export

                num_successfully_exported_components = 0
                if pose_saved_path: num_successfully_exported_components +=1
                if depth_saved_path: num_successfully_exported_components +=1
                if color_saved_path: num_successfully_exported_components +=1
                if instance_saved_path: num_successfully_exported_components +=1
                
                if num_successfully_exported_components >= min_successful_components_per_frame_export:
                    final_valid_frames_count += 1
                else:
                    print(f"Info: Frame {original_idx} did not meet min successful components ({num_successfully_exported_components}/{min_successful_components_per_frame_export}). Cleaning up.")
                    if pose_saved_path and os.path.exists(pose_saved_path): os.remove(pose_saved_path)
                    if depth_saved_path and os.path.exists(depth_saved_path): os.remove(depth_saved_path)
                    if color_saved_path and os.path.exists(color_saved_path): os.remove(color_saved_path)
                    if instance_saved_path and os.path.exists(instance_saved_path): os.remove(instance_saved_path)

            except IndexError: 
                print(f"Warning: Frame index {original_idx} out of bounds for scene {scene_id} during export. Critical error.")
                break 
            except Exception as e:
                print(f"Warning: Unhandled error processing selected frame {original_idx} for scene {scene_id}: {e}. Skipping frame.")
                total_skipped_components_in_export += 4 
                processed_attempt_count += 1
                continue
    finally:
        if instance_zip:
            instance_zip.close()
            print(f"Closed instance mask zip for {scene_id}")

    scene_export_successful = False
    if total_raw_frames == 0:
        scene_export_successful = True 
    elif processed_attempt_count > 0 and final_valid_frames_count >= min_valid_frames_per_scene:
        scene_export_successful = True
    elif not actual_indices_to_process and total_raw_frames > 0 : 
        scene_export_successful = False 
    else: 
        scene_export_successful = False
        
    if scene_export_successful:
        print(f'Finished exporting for scene {scene_id} (split: {split}).')
        print(f'Initial valid candidates: {num_candidates}. Selected for processing: {len(actual_indices_to_process)}.')
        print(f'Attempted export for {processed_attempt_count} selected frames.')
        print(f'Number of finally valid frames (>= {min_successful_components_per_frame_export} components exported): {final_valid_frames_count} (required >= {min_valid_frames_per_scene}).')
        if total_skipped_components_in_export > 0:
            print(f'Skipped {total_skipped_components_in_export} components during export phase.')
        return scene_id
    else:
        print(f"Failed to export scene {scene_id} (split: {split}).")
        print(f"  Initial valid candidates: {num_candidates}. Selected for processing: {len(actual_indices_to_process)}.")
        print(f"  Attempted export for {processed_attempt_count} selected frames.")
        print(f"  Number of finally valid frames (>= {min_successful_components_per_frame_export} components exported): {final_valid_frames_count} (required >= {min_valid_frames_per_scene}).")
        if not valid_frame_candidates_info and total_raw_frames > 0:
            print("  Reason: No frame candidates passed initial validation.")
        elif not actual_indices_to_process and valid_frame_candidates_info:
             print("  Reason: No frames were selected from candidates for processing.")
        elif final_valid_frames_count < min_valid_frames_per_scene and processed_attempt_count > 0 :
             print(f"  Reason: Number of finally valid frames ({final_valid_frames_count}) is less than the required minimum ({min_valid_frames_per_scene}).")
        return None

def main():
    # export single scene to test 
    sens_file_path = '/data/Datasets/ScanNet/scans/scene0000_01/scene0000_01.sens'
    output_base_dir = '/data/Datasets/ScanNet/vsi-split/test/01'
    num_frames_to_sample = 32
    split = 'train'
    image_size = (480, 640)
    min_valid_components_per_frame_initial = 0
    min_valid_frames_per_scene = 1
    export_scene_sampled_frames(sens_file_path, output_base_dir, num_frames_to_sample, split, image_size, min_valid_components_per_frame_initial, min_valid_frames_per_scene)

if __name__ == '__main__':
    main() 