import os
import argparse
import numpy as np
import shutil
from XFDing import progressbar2 as progressbar
from datetime import datetime
starttime = datetime.now()

# Path to directory containing TIFF images
parser = argparse.ArgumentParser(description='Reduce the number of files in tomo folder')
parser.add_argument('-PATH', type=str, required = True, help='Path to tomo folder')
parser.add_argument('-SAVE', type=str, required = True, help='Path to saving reduced tomo folder')
parser.add_argument('-reduceby', type=int, default=2, help='reduce to what number')
args, unparsed = parser.parse_known_args()

PATH = args.PATH
SAVE = args.SAVE
reduceby = args.reduceby

# for i in range(np.size(tomodirsub)):
if os.path.isdir(PATH):

	# get number of files inside the desired directory
	files = sorted([f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH,f))])
	filesnum = np.size(files)

	# copy files
	if not os.path.isdir(SAVE):	
		os.mkdir(SAVE)
	for j in progressbar(np.arange(0,filesnum,reduceby),  "Copying projections: "):
		# print(files[j])
		shutil.copy(os.path.join(PATH,files[j]), os.path.join(SAVE,files[j]))

# otherwise notify that raw data is incomplete
else:
	print("Incomplete raw data folder")

elapsedtime = str(datetime.now() - starttime)
print("Finisehd in " + elapsedtime)
starttime = datetime.now()

