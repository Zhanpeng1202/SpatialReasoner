#!/bin/bash

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="Journey9ni/vlm-3r-llava-qwen2-lora"
CONV_MODE="qwen_1_5"
FRAMES=64
POOL_STRIDE=2
POOL_MODE=average
NEWLINE_POSITION=grid
OVERWRITE=True
VIDEO_PATH="playground/demo/47334096.mp4"
MODEL_BASE="lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
CUDA_VISIBLE_DEVICES=7 python3 playground/demo/video_demo.py \
    --model-path $CKPT \
    --model-base ${MODEL_BASE} \
    --video_path ${VIDEO_PATH} \
    --output_dir ./work_dirs/video_demo/$SAVE_DIR \
    --output_name pred \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --prompt "If I am standing by the stool and facing the stove, is the sofa to my left, right, or back?\nAn object is to my back if I would have to turn at least 135 degrees in order to face it."