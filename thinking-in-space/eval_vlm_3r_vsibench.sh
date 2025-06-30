export CUDA_VISIBLE_DEVICES=3,4,5,6,7 # If you have multiple GPUs, you can set the actual GPU IDs, e.g., "0,1,2"
export LMMS_EVAL_LAUNCHER="accelerate"

# If you have multiple GPUs, you can set --num_processes=the number of GPUs to use

accelerate launch \
    --num_processes=5 \
    -m lmms_eval \
    --model vlm_3r_tstar \
    --model_args pretrained=/data/zhanpeng/weight/vlm3r/vlm-3r-llava-qwen2-lora,model_base=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32 \
    --tasks vsibench \
    --batch_size 60 \
    --log_samples \
    --log_samples_suffix vlm_3r_7b_qwen2_lora \
    --output_path logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench \
