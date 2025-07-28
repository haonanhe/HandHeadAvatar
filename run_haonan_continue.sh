cd /home/haonan/data/HHAvatar/code

eval "$(conda shell.bash hook)"
conda activate flare
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare

export CUDA_VISIBLE_DEVICES=7

# python scripts/exp_runner.py --conf ./confs/final/haonan.conf --wandb_workspace HHAvatar --nepoch 50

# python scripts/exp_runner.py --conf ./confs/final/haonan.conf --wandb_workspace HHAvatar --nepoch 360 --checkpoint 50 --is_continue --load_path /home/haonan/data/HHAvatar/data/experiments/haonan/haonan_headonly_depthw1_flamedistw5000_rgbw1_nocloth/rgb/train

# python scripts/exp_runner.py --conf ./confs/final/haonan.conf --wandb_workspace HHAvatar --nepoch 540 --checkpoint 320 --is_continue --load_path /home/haonan/data/HHAvatar/data/experiments/haonan/haonan_headonly_depthw1_flamedistw5000_rgbw1_nocloth_contactonly_20_400000_sample4_continue_30_400000/rgb/train

python scripts/exp_runner.py --conf ./confs/final/haonan.conf --wandb_workspace HHAvatar --nepoch 540 --is_continue

# render
python scripts/exp_runner.py --conf /home/haonan/data/HHAvatar/data/experiments/haonan/haonan_headonly_depthw1_flamedistw5000_rgbw1_nocloth_contactonly_20_400000_sample4_continue_30_400000_continue_100_400000/rgb/train/runconf.conf --is_eval --checkpoint latest

# create video
root_dir="/home/haonan/data/HHAvatar/data/experiments"
video_name="haonan"
epoch="epoch_540"
clip_name="rgb"
exp_name="/haonan_headonly_depthw1_flamedistw5000_rgbw1_nocloth_contactonly_20_400000_sample4_continue_30_400000_continue_100_400000"
cd $root_dir/$video_name/$exp_name/$clip_name/"eval"/$clip_name/$epoch

echo $pwd
eval "$(conda shell.bash hook)"
conda activate flare
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare

ffmpeg -framerate 10 -pattern_type glob -i './normal_head/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ./normal_head.mp4
ffmpeg -framerate 10 -pattern_type glob -i './normal_all/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ./normal_all.mp4
ffmpeg -framerate 10 -pattern_type glob -i './rgb_all/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ./rgb.mp4 
ffmpeg -framerate 10 -pattern_type glob -i './rgb_gt/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ./rgb_gt.mp4 