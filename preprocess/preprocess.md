# Capturing Head Avatar with Hand Contacts from a Monocular Video

## Capture
For subject:
* Clothing color ≠ background color
* Slow and controlled hand and head movement to minimize motion blur
* Make sure more fingers are visible, to ensure HaMer is accurate.

For camera:
* Static camera, static lighting during the entire capture
* Bright, scattered, yet directional lighting works the best, e.g. light coming from a window on the side.
* Put the video(s) in a folder for the next steps. Smoothness will be encouraged for each video, so don’t concatenate separate videos.
## Preprocess
Acknowledgement: this preprocessing pipeline uses the following submodules:
* FLAME parameter estimation from [DECA](https://github.com/YadiraF/DECA)
* MANO parameter estimation from [HaMer](https://github.com/geopavlakos/hamer?tab=readme-ov-file)
* Background segmentation with [ModNet](https://github.com/ZHKKKe/MODNet)
* Face Landmark from [face alignment](https://github.com/1adrianb/face-alignment)
* Hand Landmark from [Sapiens](https://github.com/facebookresearch/sapiens)
* Semantic segmentation from [face parser](https://github.com/zllrunning/face-parsing.PyTorch) (Optional, not in the original paper) 
* Iris estimation from [face-detection-tflite](https://github.com/patlevin/face-detection-tflite) (Optional, not in the original paper, code adapted from [neural-head-avatar](https://github.com/philgras/neural-head-avatars/))
* Hand and face segmentation from [Cutie](https://github.com/hkchengrex/Cutie)
* Depth estimation from [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
### Step 1: Getting started
#### Install environment
* Install ffmpeg: `sudo apt install ffmpeg`
* Navigate to the preprocess folder: `cd HandHeadAvatar/preprocess`
* Install [DECA](https://github.com/YadiraF/DECA) environment `bash ./submodules/DECA/install_conda.sh` and activate `conda activate deca-env`.
* If using iris tracking, `pip install -U face-detection-tflite`
#### Add some files to the submodules
* Swap in some files:
```angular2html
rsync -avh ./DECA/ ./submodules/DECA/
rsync -avh ./face-parsing.PyTorch/ ./submodules/face-parsing.PyTorch/
rsync -avh ./hamer/ ./submodules/hamer/
```
* Follow [DECA instructions](https://github.com/YadiraF/DECA) and add [generic_model.pkl (FLAME 2020)](https://flame.is.tue.mpg.de) and [deca_model.tar](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view?usp=sharing) into `./submodules/DECA/data`
* Download [modnet_webcam_portrait_matting.ckpt](https://drive.google.com/file/d/1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX/view?usp=sharing) and put into `./submodules/MODNet/pretrained/`
* If using semantic information:
  * `mkdir -p ./submodules/face-parsing.PyTorch/res/cp`
  * Download [79999_iter.pth](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) and put into `./submodules/face-parsing.PyTorch/res/cp/`

### Step 2: Run Preprocessing
* Structure your dataset as `path/subject_name/video_name`
* Your video should contain a head and a hand and cropped to a square
* Run the first session in `preprocess.sh` to get the scaled video named '${video_name}_cropped.mp4'. Use Cutie to manually segment the hand and head for the scaled video. Rename the soft masks to 'hand_mask' and 'head_mask', put them under `path/subject_name/video_name`. See more details in [Cutie](https://github.com/hkchengrex/Cutie).
* Modify `preprocess.sh` according to your data and run `bash preprocess.sh`