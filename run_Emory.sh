cd /home/haonan/data/HHAvatar/code

eval "$(conda shell.bash hook)"
conda activate flare
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare

export CUDA_VISIBLE_DEVICES=2

# python scripts/exp_runner.py --conf ./confs/final_metahuman/Emory.conf --wandb_workspace HHAvatar --nepoch 60

python scripts/exp_runner.py --conf ./confs/final_metahuman/Emory.conf --wandb_workspace HHAvatar --nepoch 360 --checkpoint 60 --is_continue --load_path /home/haonan/data/HHAvatar/data/experiments/Emory/Emory_headonly_depthw10_rgbw1_v2/rgb/train

# python scripts/exp_runner.py --conf ./confs/final_metahuman/Emory.conf --wandb_workspace HHAvatar --nepoch 360 --is_continue

# python scripts/exp_runner.py --conf ./confs/final_metahuman/Emory.conf --wandb_workspace HHAvatar --nepoch 540 --checkpoint 360 --is_continue --load_path /home/haonan/data/HHAvatar/data/experiments/Emory/Emory_headonly_depthw10_rgbw1_v2_contactonly_10_300000_sample2/rgb/train

# render
exp_name="/Emory_contactonly_20_300000_sample2"
python scripts/exp_runner.py --conf ../data/experiments/Emory/$exp_name/rgb/train/runconf.conf --is_eval --checkpoint latest

# create video
root_dir="/home/haonan/data/HHAvatar/data/experiments"
video_name="Emory"
epoch="epoch_240"
clip_name="rgb"
# exp_name="/Emory_contactonly_20_400000_sample2"
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
