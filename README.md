# Deep Learning Phase Contrast Enhancement Strategy
## Preamble
This work has been published in [https://doi.org/10.1107/S1600577523000826](https://doi.org/10.1111/jmi.13419).

## Setup Environemnets:

Create your virtual environment and install dependencies: 
  ```
python3.6 -m venv venv
source /.../venv/bin/activate
pip install --upgrade pip
pip install numpy tensorflow tensorflow_addons keras imageio scipy tifffile matplotlib
  ```

Flowchart of procedure:

![Methods](https://github.com/xfding57/EdgeView-Segmentation/blob/main/media/Figure2.jpg)

Example of Results:

![Training Loss](https://github.com/xfding57/EdgeView-Segmentation/blob/main/media/sc23-test0037-2.gif)

## Step A and B
### Run 1-split.py:

Running this script splits the original raw X-ray projection dataset. For example, if originally, 3000 X-ray projections were collected while the sample was rotated 180 degrees, with some flat-field and dark-field images, keep 3000 projection images in a "tomo" folder while the flat-field and dark-field projections are kept in "flats" and "darks" directories, respectively. 

In the example below, the raw data "tomo", "flats", and "darks" are kept in a folder called "data". The user-defined save directory isa called "datasplit". The user-defined integer number of splits is 2, the script will create two subdirecties "0000" and "0001", each with their own "tomo", "flats", and "darks" directories. The "flats" and "darks" are copied directly from "data/flats" and "data/darks" while each tomo contains 1500 projections non-overlapping.

**Command-line example:**
  ```
python 1-split.py -PATH ~/data -SAVE ~/datasplit -split 2
  ```

### Run 2-reconstruction.py:

This script is a python wrapper to run tofu-ez, for automation purposes. 

In the example below, suppose the X-ray projections were 2048 pixels width and 600 pixels height. The CoR (center of roatation) was selected to be 1024, and the user has defined to reconstruct every slice from the 1st to the 600th slice.

**Command-line example:**
  ```
python 2-reconstruction.py -PATH ~/datasplit/0000 -flatsname flats -darksname darks -tomoname tomo -SAVE ~/datasplit-rec/0000 -number 1500 -CoR 1024 -regionstart 1 -regionthickness 600
  ```

### Run 3-make-test-stacks.py:

This script processes the reconstructed 32-bit floating point CT slices. This is to adjust the histogram and save as 16-bit integer CT slices. This is because not all of the dyanmic range is necessary. There are three modes which are: 1. the entire histogram, 2. the user-defined histogram bounds plus some expansion, and 3. just the user-defined histogram bounds.

In the example below, the user defined to to use mode 1 to read the reconstructed CT images and convert from 32-bit floating point to 16-bit integers while using the entire histogram. The script also creates a test dataset called "A-test" which uses slices 200 to 210.

**Command-line example:**
  ```
python 3-make-test-stacks.py -PATH ~/datasplit-rec/0000 -SAVE ~/trainingset/A -mode 1 -teststackstart 200 -teststackthickness 10
  ```

In the example below, the user defined to to use mode 2 to read the reconstructed CT images. It then defines the boundaries to clip the histogram to contain 99.5% of pixel values while expanding above and below those bounds by 0.33 of the boundary range. It then converts from 32-bit floating point to 16-bit integers while using the entire histogram and saving to a user-defined folder "A" in "trainingset". The script also creates a test dataset called "A-test" which uses slices 200 to 210.

**Command-line example:**
  ```
python 3-make-test-stacks.py -PATH ~/datasplit-rec/0000 -SAVE ~/trainingset/A -histperc 0.995 -expansion 0.33 -mode 2 -teststackstart 200 -teststackthickness 10
  ```

### Run 4-train.py:
**Command-line example:**
  ```
python 4-train.py -PATH ~/trainingset -SAVE ~/trainingset/train -bitrate 16 weights 0,0,0,1 -learningrate 1e-4 -lmse 1 -lunet 4 -depth 3 -patchsize 64 -batchsize 32 -itg 1 -epochs 10000 -epochsave 200
  ```
### Run 5-predict.py:
**Command-line example:**
  ```
python 5-predict.py -PATH ~/trainingset/B -SAVE ~/trainingset/B-predict -NN ~/trainingset/train/test0000/it09999.h5
  ```
## Step C and D
### Threshold for rough segmentation:

### Run imagej-extract-slices.txt and nterpolate between extracted slices:

### Restore to original image size if cropped:
