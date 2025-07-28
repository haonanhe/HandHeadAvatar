cd /home/haonan/data/IMAvatar_hand_head/code

# eval "$(conda shell.bash hook)"
# conda activate flare
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate flare

eval "$(conda shell.bash hook)"
conda activate flare
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare

# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=3
# export CUDA_VISIBLE_DEVICES=4
# export CUDA_VISIBLE_DEVICES=5
export CUDA_VISIBLE_DEVICES=6
# export CUDA_VISIBLE_DEVICES=7

# python scripts/exp_runner.py --conf ./confs/rebuttal_haonan_pca_contact_detach_bignet_rd.conf --wandb_workspace IMavatar_haonan --nepoch 60
python scripts/exp_runner.py --conf ./confs/rebuttal_haonan_pca_contact_detach_bignet_rd.conf --wandb_workspace IMavatar_haonan --nepoch 120 --is_continue --load_path /home/haonan/data/IMAvatar_hand_head/data/experiments/rebuttal/haonan_2hands/IMavatar_rebuttal_haonan2hands_rd_pca_100_100000_depth1/rgb/train
# python scripts/exp_runner.py --conf ./confs/rebuttal_haonan_pca_contact_detach_bignet_rd.conf --wandb_workspace IMavatar_haonan --nepoch 160 --is_continue --load_path /home/haonan/data/IMAvatar_hand_head/data/experiments/rebuttal/haonan_2hands/IMavatar_rebuttal_haonan2hands_rd_pca_10_50000_depth1_continue_single/rgb/train