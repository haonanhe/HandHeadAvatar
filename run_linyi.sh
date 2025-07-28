cd /home/haonan/data/HHAvatar/code

eval "$(conda shell.bash hook)"
conda activate flare
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare

export CUDA_VISIBLE_DEVICES=5

# python scripts/exp_runner.py --conf ./confs/final/linyi.conf --wandb_workspace HHAvatar --nepoch 50 #--is_continue

python scripts/exp_runner.py --conf ./confs/final/linyi.conf --wandb_workspace HHAvatar --nepoch 360 --checkpoint 50 --is_continue --load_path /home/haonan/data/HHAvatar/data/experiments/linyi/linyi_headonly_depthw1_flamedistw5000_rgbw1_nocloth/rgb/train

# python scripts/exp_runner.py --conf ./confs/final/haonan.conf --wandb_workspace HHAvatar --nepoch 60 --checkpoint 30 --is_continue --load_path /home/haonan/data/HHAvatar/data/experiments/haonan/haonan_headonly_depthw1_optimcontact_flamesdfw1_flamedistw10000_contactsdf10_contactreg1000_spatialnonrigiddeformer/rgb/train

