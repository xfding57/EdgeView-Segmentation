import os
import argparse
import numpy as np
import shutil
import sys
from util import progressbar2 as progressbar
from datetime import datetime
starttime = datetime.now()

# Path to directory containing TIFF images
parser = argparse.ArgumentParser(description='Split the tomos for sub-reconstruction')
parser.add_argument('-PATH', type=str, required = True, help='Path including flats, darks, and tomo folders')
parser.add_argument('-SAVE', type=str, required = True, help='Path for saving the splitted tomo files')
parser.add_argument('-splits', type=int, default=2, help='Number of splits')
args, unparsed = parser.parse_known_args()

PATH = args.PATH
SAVE = args.SAVE
splits = args.splits

# run only if tomo darks and flats folders are present
if os.path.isdir(PATH+"/tomo")+os.path.isdir(PATH+"/flats")+os.path.isdir(PATH+"/darks") == 3:
	if not os.path.isdir(SAVE):
		os.mkdir(SAVE)

	# get number of files inside the desired directory
	files = sorted([f for f in os.listdir(os.path.join(PATH,"tomo")) if os.path.isfile(os.path.join(PATH,"tomo",f))])
	filesnum = np.size(files)

	for i in range(splits):
		# make output folder
		if not os.path.isdir(os.path.join(SAVE,str(i+1).zfill(4))):
			os.mkdir(os.path.join(SAVE,str(i+1).zfill(4)))
			os.mkdir(os.path.join(SAVE,str(i+1).zfill(4),"tomo"))
		# copy every i-th file
		# for j in np.arange(i,filesnum,splits):
		for j in progressbar(np.arange(i,filesnum,splits), "Making split #"+str(i+1)+": "):
			# print(files[j])
			shutil.copy(os.path.join(PATH,"tomo",files[j]), os.path.join(SAVE,str(i+1).zfill(4),"tomo",files[j]))
		os.system("cp -r "+PATH+"/flats "+os.path.join(SAVE,str(i+1).zfill(4),"flats"))
		os.system("cp -r "+PATH+"/darks "+os.path.join(SAVE,str(i+1).zfill(4),"darks"))

# otherwise notify that raw data is incomplete
else:
	print("Incomplete raw data folder")

elapsedtime = str(datetime.now() - starttime)
print("Finisehd in " + elapsedtime)
starttime = datetime.now()
