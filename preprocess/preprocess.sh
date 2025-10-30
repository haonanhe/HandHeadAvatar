root_dir=/home/haonan/data/HandHead/HHAvatar/preprocess
cd $root_dir
#######################################################
subject_name='zirui'
path='/home/haonan/data/HandHead/HHAvatar/data/datasets'
video_folder=$path/$subject_name
video_names='finger.mp4 fist.mp4 palm.mp4 pinch.mp4'
shape_video='finger.mp4'
ndr_dir=$ndr_root_dir/$subject_name
fps=30
resize=512

cuda_num=5
export CUDA_VISIBLE_DEVICES=$cuda_num

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
path_hamer=$(pwd)'/submodules/hamer'
path_sapiens_lite='Your/path/to/sapiens_lite'
path_sapiens_lite_host='Your/path/to/sapiens_lite_host'
path_depthanything='Your/path/to/depthanything'
########################################################
set -e

echo "crop and resize video"
cd $pwd
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  echo $video_folder/$subject_name/"${array[0]}"/"image"
  ffmpeg -y -i $video_path -vf "fps=$fps, scale=$resize:$resize" -c:v libx264 $video_folder/"${array[0]}_cropped.mp4"
done

echo "background/foreground segmentation"
cd $path_modnet
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  mkdir -p $video_folder/$subject_name/"${array[0]}"
  python -m demo.video_matting.custom.run --video $video_folder/"${array[0]}_cropped.mp4" --result-type matte --fps $fps
done

echo "split videos into images"
# sudo apt install ffmpeg
cd $pwd
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  echo $video_folder/$subject_name/"${array[0]}"/"image"
  mkdir -p $video_folder/$subject_name/"${array[0]}"/"image"
  ffmpeg -i $video_folder/"${array[0]}_cropped.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"image"/"%07d.png"
  mkdir -p $video_folder/$subject_name/"${array[0]}"/"mask"
  ffmpeg -i $video_folder/"${array[0]}_cropped_matte.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"mask"/"%07d.png"
done

# flame
echo "DECA FLAME parameter estimation"
cd $path_deca
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate deca-env
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  mkdir -p $video_folder/$subject_name/"${array[0]}"/"deca"
  pwd
  python demos/demo_reconstruct.py -i $video_folder/$subject_name/"${array[0]}"/image --savefolder $video_folder/$subject_name/"${array[0]}"/"deca" --saveCode True --saveVis False --sample_step 1  --render_orig False --rasterizer_type pytorch3d #--saveImages True
done

echo "face alignment landmark detector"
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate deca-env
cd $pwd
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  python keypoint_detector.py --path $video_folder/$subject_name/"${array[0]}"
done

### depth estimation
cd $path_sapiens
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flare
for video in $video_names
do
    video_path=$video_folder/$video
    echo $video
    IFS='.' read -r -a array <<< $video

    python run_video.py \
    --encoder vitl \
    --video-path $video_folder/"${array}"_cropped.mp4 --outdir $video_folder/$subject_name/"${array}"/depth_imgs_v2 \
    --input-size 512 --pred-only --grayscale

    ffmpeg -i $video_folder/$subject_name/"${array}"/depth_imgs_v2/"${array}"_cropped.mp4 -q:v 2 $video_folder/$subject_name/"${array}"/depth_imgs_v2/"%07d.jpg" 
done


## mano
echo "Hamer MANO parameter estimation"
cd $path_hamer
export PYOPENGL_PLATFORM=egl
# source deactivate
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate hamer
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  image_folder=$video_folder/$subject_name/"${array[0]}"/image
  output_folder="demo_out"_$subject_name
  python demo_codes_right.py \
    --img_folder $image_folder --out_folder $output_folder \
    --batch_size=48 --save_mesh --full_frame
done

echo "semantic segmentation with face parsing"
cd $path_parser
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python test.py --dspth $video_folder/$subject_name/"${array}"/image --respth $video_folder/$subject_name/"${array}"/semantic
done

echo "sapiens landmark estimation"

DATASET_ROOT=$video_folder/$subject_name
for video in $video_names
do
  cd $path_sapiens_lite
  SAPIENS_CHECKPOINT_ROOT=$path_sapiens_lite_host
  MODE='torchscript' ## original. no optimizations (slow). full precision inference.
  SAPIENS_CHECKPOINT_ROOT=$SAPIENS_CHECKPOINT_ROOT/$MODE

  echo $video
  IFS='.' read -r -a array <<< $video
  INPUT_DIR="${DATASET_ROOT}/${array[0]}"
  INPUT="${INPUT_DIR}/image"
  OUTPUT="${INPUT_DIR}/sapiens_lmk"

  eval "$(conda shell.bash hook)"
  conda activate sapiens_lite

  #--------------------------MODEL CARD---------------
  # MODEL_NAME='sapiens_0.3b'; CHECKPOINT=coming soon!
  # MODEL_NAME='sapiens_0.6b'; CHECKPOINT=coming soon!
  # MODEL_NAME='sapiens_1b'; CHECKPOINT=coming soon!
  MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2

  OUTPUT=$OUTPUT/$MODEL_NAME

  DETECTION_CONFIG_FILE='../pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
  DETECTION_CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth

  #---------------------------VISUALIZATION PARAMS--------------------------------------------------
  LINE_THICKNESS=3 ## line thickness of the skeleton
  RADIUS=6 ## keypoint radius
  KPT_THRES=0.3 ## confidence threshold

  ##-------------------------------------inference-------------------------------------
  RUN_FILE='demo/vis_pose.py'

  ## number of inference jobs per gpu, total number of gpus and gpu ids
  # JOBS_PER_GPU=1; TOTAL_GPUS=8; VALID_GPU_IDS=(0 1 2 3 4 5 6 7)
  JOBS_PER_GPU=1; TOTAL_GPUS=1; VALID_GPU_IDS=($cuda_num)

  BATCH_SIZE=8

  # Find all images and sort them, then write to a temporary text file
  IMAGE_LIST="${INPUT}/image_list.txt"
  find "${INPUT}" -type f \( -iname \*.jpg -o -iname \*.png \) | sort > "${IMAGE_LIST}"

  # Check if image list was created successfully
  if [ ! -s "${IMAGE_LIST}" ]; then
    echo "No images found. Check your input directory and permissions."
    exit 1
  fi

  # Count images and calculate the number of images per text file
  NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
  if ((TOTAL_GPUS > NUM_IMAGES / BATCH_SIZE)); then
    TOTAL_JOBS=$(( (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE))
    IMAGES_PER_FILE=$((BATCH_SIZE))
    EXTRA_IMAGES=$((NUM_IMAGES - ((TOTAL_JOBS - 1) * BATCH_SIZE)  ))
  else
    TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))
    IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
    EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))
  fi

  export TF_CPP_MIN_LOG_LEVEL=2
  echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs."

  # Divide image paths into text files for each job
  for ((i=0; i<TOTAL_JOBS; i++)); do
    TEXT_FILE="${INPUT}/image_paths_$((i+1)).txt"
    if [ $i -eq $((TOTAL_JOBS - 1)) ]; then
      # For the last text file, write all remaining image paths
      tail -n +$((IMAGES_PER_FILE * i + 1)) "${IMAGE_LIST}" > "${TEXT_FILE}"
    else
      # Write the exact number of image paths per text file
      head -n $((IMAGES_PER_FILE * (i + 1))) "${IMAGE_LIST}" | tail -n ${IMAGES_PER_FILE} > "${TEXT_FILE}"
    fi
  done

  # Run the process on the GPUs, allowing multiple jobs per GPU
  for ((i=0; i<TOTAL_JOBS; i++)); do
    GPU_ID=$((i % TOTAL_GPUS))
    CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[GPU_ID]} python ${RUN_FILE} \
      ${CHECKPOINT} \
      --num_keypoints 133 \
      --batch-size ${BATCH_SIZE} \
      --det-config ${DETECTION_CONFIG_FILE} \
      --det-checkpoint ${DETECTION_CHECKPOINT} \
      --input "${INPUT}/image_paths_$((i+1)).txt" \
      --output-root="${OUTPUT}" \
      --radius ${RADIUS} \
      --kpt-thr ${KPT_THRES} ## add & to process in background

    # Allow a short delay between starting each job to reduce system load spikes
    sleep 1
  done

  # Wait for all background processes to finish
  wait

  # Remove the image list and temporary text files
  rm "${IMAGE_LIST}"
  for ((i=0; i<TOTAL_JOBS; i++)); do
    rm "${INPUT}/image_paths_$((i+1)).txt"
  done

  echo "Processing complete."
  echo "Results saved to $OUTPUT"

done


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

for video in $video_names
do
  if [ "$shape_video" == "$video" ];
  then
    continue
  fi
  IFS='.' read -r -a array <<< $(basename $shape_video)
  shape_from=$video_folder/$subject_name/"${array}"
  IFS='.' read -r -a array <<< $(basename $video)
  echo $video
  python optimize.py --path $video_folder/$subject_name/"${array}" --shape_from $shape_from  --cx $cx --cy $cy --fx $fx --fy $fy --size $resize --conf $root_dir/'confs'/$subject_name".conf"
done


