#!/usr/bin/env python3
"""
sample_rgb_instance.py

For every ScanNet scene inside <root_dir>:
  • Read <scene_id>.sens
  • Copy the corresponding instance mask PNG from   instance-filt/{frame_idx}.png
  • Save 1-fps RGB JPEGs      →  color-sample/{frame_idx:06d}.jpg
  • Save mask PNGs            →  instance-sample/{frame_idx:06d}.png
  • NEW: write color-sample.mp4 (preview video, 6 FPS, H.264)
  • NEW: after all scenes, dump sample_counts.json summarising #kept frames
"""

import argparse, os, json, shutil, multiprocessing as mp
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# --- ScanNet helpers ----------------------------------------------------------
from preprocess.SensorData import SensorData

STEP = 24                # keep every 24th frame  (≈ 1 fps)
VIDEO_FPS = 6            # preview video speed

# ------------------------------------------------------------------------------

def make_video(color_dir: Path, video_path: Path, fps: int):
    """Stack all *.jpg in color_dir (sorted) into a .mp4"""
    imgs = sorted(color_dir.glob("*.jpg"))
    if not imgs:
        return
    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")      # widely supported H.264
    vw = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        vw.write(frame)
    vw.release()

# ------------------------------------------------------------------------------

def process_scene(args):
    scene_path, resize_hw = args
    scene_id  = scene_path.name
    
    sens_dir = Path("/data/Datasets/ScanNet/scans")
    sens_file = sens_dir/f"{scene_id}"/f"{scene_id}.sens"

    masks_dir  = scene_path / "instance-filt"

    if not sens_file.is_file():
        return scene_id, 0, f"[{scene_id}]  MISSING .sens - skipped"

    try:
        sd = SensorData(sens_file)
    except Exception as e:
        return scene_id, 0, f"[{scene_id}]  failed to read .sens: {e}"

    out_color_dir    = scene_path / "color-sample"
    out_instance_dir = scene_path / "instance-sample"
    out_color_dir.mkdir(exist_ok=True)
    out_instance_dir.mkdir(exist_ok=True)

    frames_to_keep = range(0, len(sd.frames), STEP)
    kept = 0
    for f_idx in frames_to_keep:
        # --- RGB ----------------------------------------------------------------
        rgb = sd.frames[f_idx].decompress_color(sd.color_compression_type)
        if resize_hw is not None:
            rgb = cv2.resize(rgb, (resize_hw[1], resize_hw[0]),
                             interpolation=cv2.INTER_NEAREST)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb_path = out_color_dir / f"{f_idx:06d}.jpg"
        cv2.imwrite(str(rgb_path), rgb_bgr)

        # --- instance mask ------------------------------------------------------
        mask_src = masks_dir / f"{f_idx}.png"
        mask_dst = out_instance_dir / f"{f_idx:06d}.png"
        if mask_src.is_file():
            if resize_hw is None:
                shutil.copy(mask_src, mask_dst)
            else:
                mask = cv2.imread(str(mask_src), cv2.IMREAD_UNCHANGED)
                mask = cv2.resize(mask, (resize_hw[1], resize_hw[0]),
                                  interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(mask_dst), mask)
        kept += 1

    # --- preview video ----------------------------------------------------------
    video_path = scene_path / "color-sample.mp4"
    make_video(out_color_dir, video_path, VIDEO_FPS)

    return scene_id, kept, f"[{scene_id}]  saved {kept} frames → {video_path.name}"

# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", default="/data/Datasets/ScanNet/qa_subset_test",
                    help="Folder that contains sceneXXXX_YY sub-dirs")
    ap.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"),
                    default=(480, 640),
                    help="Resize every RGB / mask to (H,W).  Supply nothing to keep native size")
    ap.add_argument("--workers", type=int,
                    default=max(64, os.cpu_count() // 4),
                    help="Processes to run in parallel (one per scene)")
    args = ap.parse_args()

    scene_dirs = [p for p in Path(args.root_dir).iterdir() if p.is_dir()]
    print(f"Found {len(scene_dirs)} scenes under {args.root_dir}")

    sample_counts = {}           # scene_id → #frames kept
    with mp.Pool(processes=args.workers) as pool:
        for scene_id, kept, msg in tqdm(
                pool.imap_unordered(process_scene,
                                    [(p, args.resize) for p in scene_dirs]),
                total=len(scene_dirs),
                desc="Sampling scenes"):
            print(msg)
            sample_counts[scene_id] = kept

    # ---- dump global JSON ------------------------------------------------------
    out_json = Path(args.root_dir) / "sample_counts.json"
    with open(out_json, "w") as f:
        json.dump(sample_counts, f, indent=2)
    print(f"\nWrote per-scene frame counts to {out_json}")

if __name__ == "__main__":
    main()
