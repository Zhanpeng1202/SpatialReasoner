python -m src.metadata_generation.ScanNet.preprocess.export_sampled_frames \
    --scans_dir /data/Datasets/ScanNet/scans \
    --output_dir /data/Datasets/ScanNet/vlm3r\
    --train_val_splits_path /data/Datasets/ScanNet/vlm3r/trainSplit \
    --num_frames 32 \
    --max_workers 64 \
    --image_size 480 640