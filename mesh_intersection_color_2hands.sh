cd /home/haonan/data/IMAvatar_hand_head/

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
export CUDA_VISIBLE_DEVICES=5
# export CUDA_VISIBLE_DEVICES=6
# export CUDA_VISIBLE_DEVICES=7

python /home/haonan/data/IMAvatar_hand_head/mesh_intersection_color_2hands.py