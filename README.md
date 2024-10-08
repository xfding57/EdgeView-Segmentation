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
Running this script splits the original raw X-ray projection dataset. For example, if originally, 3000 projections were collected while the sample was rotated 180 degrees, then keep 3000 projection images in a "tomo" folder. While keeping flat-field and dark-field projections in "flats" and "darks" directories. If the user-defined integer number of splits is 2, then in the user-defined SAVE directory, it will create folders 0000 and 0001, each with their own tomo, flats, and darks directories. The flats and darks are copied from the original while each tomo contains 1500 projections non-overlapping.
  ```
python 1-split.py -PATH /path/to/directory/containing/tomo/flats/and/darks -SAVE /path/to/desired/directory/to/save/split/raw/datasets -split 2
  ```

### Run 2-reconstruction.py:

### Run 3-make-test-stacks.py:

### Run 4-train.py:

### Run 5-predict.py:

## Step C and D
### Threshold for rough segmentation:

### Run imagej-extract-slices.txt and nterpolate between extracted slices:

### Restore to original image size if cropped: