## Setup Environemnets:

Create your virtual environment and install dependencies: 
  ```
python3.6 -m venv venv
source /.../venv/bin/activate
pip install --upgrade pip
pip install numpy tensorflow tensorflow_addons keras imageio scipy tifffile matplotlib
  ```

Flowchart of procedure:

![Make Subfolders](https://github.com/xfding57/EdgeView-Segmentation/blob/main/media/Figure2.jpg)

## Step A and B
### Run 1-split.py:

### Run 2-reconstruction.py:

### Run 3-make-test-stacks.py:

### Run 4-train.py:

### Run 5-predict.py:

## Step C and D
### Threshold for rough segmentation:

### Run imagej-extract-slices.txt and nterpolate between extracted slices:

### Restore to original image size if cropped: