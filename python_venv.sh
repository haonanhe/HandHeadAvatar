conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate HandHeadAvatar

pip install -U pip
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
git clone https://github.com/YuliangXiu/bvh-distance-queries.git
cd bvh-distance-queries
# export CUDA_SAMPLES_INC=~/NVIDIA_CUDA-10.0_Samples/common/inc/
pip install -r requirements.txt 
python setup.py install

pip install smplx
pip install human_body_prior
rm -fr homogenus
git clone https://github.com/nghorbani/homogenus.git
cd homogenus
python setup.py install
cd ..
rm torch-mesh-isect -fr
git clone https://github.com/vchoutas/torch-mesh-isect
cd torch-mesh-isect/
export CUDA_SAMPLES_INC=~/NVIDIA_CUDA-10.0_Samples/common/inc/
python setup.py install
pip install trimesh
pip install pyrender
pip install pyaml
pip install tqdm
pip install configargparse
pip install shapely