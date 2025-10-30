# HandHeadAvatar
The code is under testing...

# Capturing Head Avatar with Hand Contacts from a Monocular Video
## [Paper](https://arxiv.org/abs/2510.17181) | [Video Youtube](https://www.youtube.com/watch?v=PgP1svuc4rI) | [Project Page](https://haonanhe.github.io/hand_head_avatar/)


Official Repository for ICCV 2025 paper [*Capturing Head Avatar with Hand Contacts from a Monocular Video*](https://arxiv.org/abs/2510.17181). 

## Getting Started
* Clone this repo: `git clone --recursive git@github.com:haonanhe/HandHeadAvatar.git`
* Create a conda environment `conda env create -f environment.yml` and activate `conda activate HandHeadAvatar` 
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

