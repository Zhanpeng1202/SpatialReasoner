import os 
from datasets import load_dataset
from pathlib import Path
import re





vsi_bench = load_dataset("/data/Datasets/VSI-Bench")
# print(vsi_bench)

vsi_dir = "/data/zhanpeng/datasets/tstar/vsi_tstar_plain"

dataset_name = vsi_bench["test"][1]['dataset']
scene_name   = vsi_bench["test"][1]['scene_name']

video_path = os.path.join(vsi_dir, dataset_name, f"{scene_name}.mp4")
question   = vsi_bench["test"][1]['question']
options    = vsi_bench["test"][1]['options']
ground_truth = vsi_bench["test"][1]['ground_truth']

tstar_dir = os.path.join(vsi_dir, os.path.basename(video_path).split('.')[0], question[:-1])
frames_paths = Path(os.path.join(tstar_dir, "frames"))
frame_info  = []

for img_path in list(frames_paths.glob("*.jpg")):
    # Pattern: frame_{number}_at_{timestamp}s.jpg
    
    pattern = r"frame_(\d+)_at_([\d.]+)s\.jpg"
    match   = re.match(pattern, img_path.name)
    if match:
        frame_num = int(match.group(1))
        timestamp = float(match.group(2))
        # print(f"Frame {frame_num} at {timestamp} seconds")
        
        frame_info.append((frame_num, timestamp, img_path))
    else:
        print(f"No match found for {img_path.name}")

frame_info.sort(key=lambda x: x[0])

for frame_num, timestamp, img_path in frame_info:
    print(frame_num, timestamp, img_path)







