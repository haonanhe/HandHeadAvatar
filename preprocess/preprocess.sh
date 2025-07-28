root_dir=/home/haonan/data/HHAvatar/preprocess
cd $root_dir
#######################################################
# subject_name='haonan'
# path='/home/haonan/data/HHAvatar/data/datasets'
# ndr_root_dir='/home/haonan/data/NDR_datasets'
# video_folder=$path/$subject_name
# video_names='rgb.mp4'
# shape_video='rgb.mp4'
# depth_video_names='depth.mp4'
# mask_video_names='head_mask.mp4 hand_mask.mp4'
# ndr_dir=$ndr_root_dir/$subject_name
# fps=30
# resize=512
# crop="500:500:100:300"

subject_name='linyi'
path='/home/haonan/data/HHAvatar/data/datasets'
ndr_root_dir='/home/haonan/data/NDR_datasets'
video_folder=$path/$subject_name
video_names='rgb.mp4'
shape_video='rgb.mp4'
depth_video_names='depth.mp4'
mask_video_names='head_mask.mp4 hand_mask.mp4'
ndr_dir=$ndr_root_dir/$subject_name
fps=30
resize=512
crop="580:580:80:300"

# subject_name='luocheng'
# path='/home/haonan/data/HHAvatar/data/datasets'
# ndr_root_dir='/home/haonan/data/NDR_datasets'
# video_folder=$path/$subject_name
# video_names='rgb.mp4'
# shape_video='rgb.mp4'
# depth_video_names='depth.mp4'
# mask_video_names='head_mask.mp4 hand_mask.mp4'
# ndr_dir=$ndr_root_dir/$subject_name
# fps=30
# resize=512
# crop="580:580:80:200"


# subject_name='zirui'
# path='/home/haonan/data/HHAvatar/data/datasets'
# ndr_root_dir='/home/haonan/data/NDR_datasets'
# video_folder=$path/$subject_name
# video_names='rgb.mp4'
# shape_video='rgb.mp4'
# depth_video_names='depth.mp4'
# mask_video_names='head_mask.mp4 hand_mask.mp4'
# ndr_dir=$ndr_root_dir/$subject_name
# fps=30
# resize=512
# crop="640:640:100:160"


export CUDA_VISIBLE_DEVICES=0


eval "$(conda shell.bash hook)"
conda activate flare
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare

# fx, fy, cx, cy in pixels, need to adjust with resizing and cropping
fx=1539.67462
fy=1508.93280
cx=261.442628
cy=253.231895
########################################################
pwd=$(pwd)
path_modnet=$(pwd)'/submodules/MODNet'
path_deca=$(pwd)'/submodules/DECA'
path_parser=$(pwd)'/submodules/face-parsing.PyTorch'
########################################################
set -e

# echo "crop and resize video"
# cd $pwd
# for video in $video_names
# do
#   video_path=$video_folder/$video
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   echo $video_folder/$subject_name/"${array[0]}"/"image"
#   ffmpeg -y -i $video_path -vf "fps=$fps, crop=$crop, scale=$resize:$resize" -c:v libx264 $video_folder/"${array[0]}_cropped.mp4"
# done

# for video in $depth_video_names
# do
#   video_path=$video_folder/$video
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   ffmpeg -y -i $video_path -vf "fps=$fps, crop=$crop, scale=$resize:$resize" -c:v libx264 $video_folder/"${array[0]}_cropped.mp4"
# done

# for video in $mask_video_names
# do
#   video_path=$video_folder/$video
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   ffmpeg -y -i $video_path -vf "fps=$fps, crop=$crop, scale=$resize:$resize" -c:v libx264 $video_folder/"${array[0]}_cropped.mp4"
# done

# echo "background/foreground segmentation"
# cd $path_modnet
# for video in $video_names
# do
#   video_path=$video_folder/$video
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   mkdir -p $video_folder/$subject_name/"${array[0]}"
#   python -m demo.video_matting.custom.run --video $video_folder/"${array[0]}_cropped.mp4" --result-type matte --fps $fps
# done

# echo "split videos into images"
# # sudo apt install ffmpeg
# cd $pwd
# for video in $video_names
# do
#   video_path=$video_folder/$video
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   echo $video_folder/$subject_name/"${array[0]}"/"image"
#   mkdir -p $video_folder/$subject_name/"${array[0]}"/"image"
#   ffmpeg -i $video_folder/"${array[0]}_cropped.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"image"/"%07d.png"
#   mkdir -p $video_folder/$subject_name/"${array[0]}"/"mask"
#   ffmpeg -i $video_folder/"${array[0]}_cropped_matte.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"mask"/"%07d.png"
#   mkdir -p $video_folder/$subject_name/"${array[0]}"/"hand_mask"
#   ffmpeg -i $video_folder/"hand_mask_cropped.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"hand_mask"/"%07d.png"
#   mkdir -p $video_folder/$subject_name/"${array[0]}"/"head_mask"
#   ffmpeg -i $video_folder/"head_mask_cropped.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"head_mask"/"%07d.png"
#   mkdir -p $video_folder/$subject_name/"${array[0]}"/"depth"
#   ffmpeg -i $video_folder/"depth_cropped.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"depth"/"%07d.png"
# done

# # flame
# echo "DECA FLAME parameter estimation"
# cd $path_deca
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate deca-env
# for video in $video_names
# do
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   mkdir -p $video_folder/$subject_name/"${array[0]}"/"deca"
#   python demos/demo_reconstruct.py -i $video_folder/$subject_name/"${array[0]}"/image --savefolder $video_folder/$subject_name/"${array[0]}"/"deca" --saveCode True --saveVis False --sample_step 1  --render_orig False --rasterizer_type pytorch3d #--saveImages True
# done

# echo "face alignment landmark detector"
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate deca-env
# cd $pwd
# for video in $video_names
# do
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   python keypoint_detector.py --path $video_folder/$subject_name/"${array[0]}"
# done


# ### depth estimation
# cd /home/haonan/data/Depth-Anything-V2
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate flare
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate flare
# for video in $video_names
# do
#     video_path=$video_folder/$video
#     echo $video
#     IFS='.' read -r -a array <<< $video

#     python run_video.py \
#     --encoder vitl \
#     --video-path $video_folder/"${array}"_cropped.mp4 --outdir $video_folder/$subject_name/"${array}"/depth_imgs_v2 \
#     --input-size 512 --pred-only --grayscale

#     ffmpeg -i $video_folder/$subject_name/"${array}"/depth_imgs_v2/"${array}"_cropped.mp4 -q:v 2 $video_folder/$subject_name/"${array}"/depth_imgs_v2/"%07d.jpg" 
# done


# ## mano
# echo "Hamer MANO parameter estimation"
# cd /home/haonan/Codes/hamer
# export PYOPENGL_PLATFORM=egl
# # source deactivate
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate hamer
# for video in $video_names
# do
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   # mkdir -p $video_folder/$subject_name/"${array[0]}"/"deca"
#   image_folder=$video_folder/$subject_name/"${array[0]}"/image
#   output_folder="demo_out"_$subject_name
#   python demo_codes_right.py \
#     --img_folder $image_folder --out_folder $output_folder \
#     --batch_size=48 --save_mesh --full_frame
#   # python demos/demo_reconstruct.py -i $video_folder/$subject_name/"${array[0]}"/image --savefolder $video_folder/$subject_name/"${array[0]}"/"deca" --saveCode True --saveVis False --sample_step 1  --render_orig False --rasterizer_type pytorch3d #--saveImages True
# done


echo "fit MANO & FLAME parameter for one video: "$shape_video
cd $pwd
echo $pwd
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare
IFS='.' read -r -a array <<< $shape_video
python ./optimize.py --path $video_folder/$subject_name/"${array}" --cx $cx --cy $cy --fx $fx --fy $fy --size $resize --conf $root_dir/'confs'/$subject_name".conf"

