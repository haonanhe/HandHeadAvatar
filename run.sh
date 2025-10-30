#!/bin/bash

# Automatically set the project root directory (assumes script is run from the project root)
ROOT_DIR=$(pwd)

# Navigate to the code directory (relative path)
cd "$ROOT_DIR/code"

# Initialize conda and activate the HandHeadAvatar environment
eval "$(conda shell.bash hook)"
conda activate HandHeadAvatar

# Set visible GPU device(s)
export CUDA_VISIBLE_DEVICES=0

# Configuration parameters
conf_name="zirui"
video_name="zirui"
exp_name="/zirui"  # Keep leading slash to match original path structure
video_names='finger.mp4 fist.mp4 palm.mp4 pinch.mp4'
# Remove .mp4 extensions and join with '+' for train_names
train_names=$(echo "$video_names" | sed 's/\.mp4//g' | tr ' ' '+')

# Stage 1: Reconstruct the neural head avatar
python scripts/exp_runner.py --conf "./confs/${conf_name}.conf" --wandb_workspace HHAvatar --nepoch 50

# Stage 2: Fine-tune non-rigid deformations with contact optimization
# Note: Ensure that ./confs/${conf_name}.conf has `optimize_contact` and `contact_only` set to True
python scripts/exp_runner.py \
    --conf "./confs/${conf_name}.conf" \
    --wandb_workspace HHAvatar \
    --nepoch 250 \
    --checkpoint 50 \
    --is_continue \
    --load_path "$ROOT_DIR/data/experiments/${video_name}${exp_name}/${train_names}/train"

# Stage 3: Render results using the latest checkpoint
python scripts/exp_runner.py \
    --conf "$ROOT_DIR/data/experiments/${video_name}${exp_name}/${train_names}/train/runconf.conf" \
    --is_eval \
    --checkpoint latest

# Stage 4: Generate videos from rendered image sequences
epoch="epoch_250"
clip_name="$train_names"
# Extract base names without .mp4
sub_clip_name=$(echo "$video_names" | sed 's/\.mp4//g')

# Switch to the 'flare' conda environment (assumed to contain ffmpeg or other required tools)
conda activate flare

# Loop over each clip to generate MP4 videos
for video in $sub_clip_name; do
    # Construct the evaluation output directory path
    eval_dir="$ROOT_DIR/data/experiments/${video_name}${exp_name}/${clip_name}/eval/${video}/${epoch}"
    
    echo "Processing: $eval_dir"
    
    # Skip if the evaluation directory does not exist
    if [ ! -d "$eval_dir" ]; then
        echo "Warning: $eval_dir does not exist, skipping."
        continue
    fi

    # Change to the evaluation directory
    cd "$eval_dir"

    # Encode image sequences into MP4 videos with proper padding for even dimensions
    ffmpeg -framerate 10 -pattern_type glob -i './normal_head/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y ./normal_head.mp4
    ffmpeg -framerate 10 -pattern_type glob -i './normal_all/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y ./normal_all.mp4
    ffmpeg -framerate 10 -pattern_type glob -i './rgb_all/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y ./rgb.mp4
    ffmpeg -framerate 10 -pattern_type glob -i './rgb_gt/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y ./rgb_gt.mp4
done

echo "Pipeline completed successfully!"