# HandHeadAvatar
The code is under testing...

# Capturing Head Avatar with Hand Contacts from a Monocular Video
## [Paper](https://arxiv.org/abs/2510.17181) | [Video Youtube](https://www.youtube.com/watch?v=PgP1svuc4rI) | [Project Page](https://haonanhe.github.io/hand_head_avatar/)


Official Repository for ICCV 2025 paper [*Capturing Head Avatar with Hand Contacts from a Monocular Video*](https://arxiv.org/abs/2510.17181). 

## Getting Started
* Clone this repo: `git clone --recursive git@github.com:haonanhe/HandHeadAvatar.git`
* Set up the environment

    ```bash
      conda create -n HandHeadAvatar python=3.9
      conda activate HandHeadAvatar
      conda install -c fvcore -c iopath -c conda-forge fvcore iopath
      conda install -c bottler nvidiacub
      conda install -c conda-forge ffmpeg
      pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

      wget https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.3.zip
      unzip v0.7.3.zip
      cd pytorch3d-0.7.3
      python3 setup.py install
      cd ..

      pip install ninja imageio PyOpenGL glfw xatlas gdown
      pip install git+https://github.com/NVlabs/nvdiffrast/
      export TCNN_CUDA_ARCHITECTURES="70;75;80" 
      export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc-9'
      pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
      imageio_download_bin freeimage

      pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.0_cu117/kaolin-0.15.0-cp39-cp39-linux_x86_64.whl
      
      pip install -r requirements.txt
    ```
* We use `libmise` to extract 3D meshes, build `libmise` by running `cd code; python setup.py install`
* Download [FLAME model](https://flame.is.tue.mpg.de/download.php), choose **FLAME 2020** and unzip it, copy 'generic_model.pkl' into `./code/flame/FLAME2020`
* Download [MANO model](https://mano.is.tue.mpg.de/download.php), choose **Models & Code** and unzip it, copy 'MANO_RIGHT.pkl' into `./code/mano_model/data/mano`
## Preparing dataset
prepare your own dataset following intructions in `./preprocess/preprocess.md`.

Link the dataset folder to `./data/datasets`. Link the experiment output folder to `./data/experiments`.

## Training
```
bash run.sh
```

## Citation
If you find our code or paper useful, please cite as:
```
@misc{he2025capturingheadavatarhand,
      title={Capturing Head Avatar with Hand Contacts from a Monocular Video}, 
      author={Haonan He and Yufeng Zheng and Jie Song},
      year={2025},
      eprint={2510.17181},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.17181}, 
}
```

