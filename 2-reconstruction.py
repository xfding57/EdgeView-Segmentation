import os, re, subprocess, tifffile
import numpy as np
from shutil import copyfile
from datetime import datetime
from imageio import get_writer
from PIL import Image
import argparse
import util2
run = 1
starttime = datetime.now()
present_dir = os.getcwd()

# Set up argparse to accept command line arguments
parser = argparse.ArgumentParser(description='Wrapper for CT reconstruction using ')
parser.add_argument('-PATH', type=str, default="", help='Path for the results')
parser.add_argument('-flatsname', type=str, default='flats', help='Name of flats folder')
parser.add_argument('-darksname', type=str, default='darks', help='Name of darks folder')
parser.add_argument('-tomoname', type=str, default='tomo', help='Name of tomo folder')
parser.add_argument('-SAVE', type=str, default="", help='Path for save reconstructed results')
parser.add_argument('-TEMP', type=str, default="", help='Path for temporary folders')
parser.add_argument('-number', type=int, default=3000, help='Number of projections in one 180 scan')
parser.add_argument('-CoR', type=float, default=1012, help='Center of Rotation')
parser.add_argument('-regionstart', type=int, default=300, help='Vertical position in projection for start of reconstruction')
parser.add_argument('-regionthickness', type=int, default=1, help='How many slices to reconstruct')
parser.add_argument('-cropmode', type=int, default=0, help='0 or 1')
parser.add_argument('-xcropstart', type=int, default=0, help='')
parser.add_argument('-xcroplength', type=int, default=2048, help='')
parser.add_argument('-ycropstart', type=int, default=0, help='')
parser.add_argument('-ycroplength', type=int, default=2048, help='')
parser.add_argument('-rotate', type=float, default=2048, help='')
parser.add_argument('-phasemode', type=int, default=0, help='Use phase retreival (0 or 1)')
parser.add_argument('-energy', type=float, default=30, help='X-ray energy of monochromatic beam or peak energy of white beam')
parser.add_argument('-distance', type=float, default=1.5, help='Distance between sample and detector')
parser.add_argument('-pixelsize', type=float, default=13e-6, help='Effective pixel size')
parser.add_argument('-deltabeta', type=float, default=2000, help='Delta over beta ratio')
parser.add_argument('-ringremovalmode', type=int, default=0, help='Use ring removal (0 or 1)')
parser.add_argument('-sigmah', type=int, default=11, help='Horizontal sigma')
parser.add_argument('-sigmav', type=int, default=1, help='Vertical sigma')
args = parser.parse_args()

################## INPUTS ##################

# relavant paths
PATH = args.PATH
flatsname = args.flatsname
darksname = args.darksname
tomoname = args.tomoname
SAVE = args.SAVE
TEMP = args.TEMP
# Remove temporary folder if it exists
if os.path.isdir(TEMP):
	os.system("rm -rf "+TEMP)

# projection values
number = args.number
width, height = util2.get_width_height(os.path.join(PATH,tomoname))
CoR = args.CoR
region_start = args.regionstart
region_thick = args.regionthickness

# reconstruction region
if region_start+region_thick > height:
	print("Exceeds spatial region")
	quit()
region_all = np.arange(np.ceil(-height/2),np.ceil(height/2+1),1)
region_some = np.arange(np.ceil(-region_thick/2),np.ceil(region_thick/2+1),1)
if args.ringremovalmode == 0:
	regioncommand = "--region="+str(region_all[region_start])+","+str(region_all[region_start+region_thick])+",1 "
	if args.cropmode == 1:
		regioncommand = "--region="+str(region_all[region_start])+","+str(region_all[region_start+region_thick])+",1 --x-region="+str(args.xcropstart-args.xcroplength)+","+str(args.xcropstart)+",1 --y-region="+str(args.ycropstart-args.ycroplength)+","+str(args.ycropstart)+",1 "
elif args.ringremovalmode == 1:
	regioncommand = "--region="+str(region_some[0])+","+str(region_some[len(region_some)-1])+",1 "
	if args.cropmode == 1:
		regioncommand = "--region="+str(region_some[0])+","+str(region_some[len(region_some)-1])+",1 --x-region="+str(args.xcropstart-args.xcroplength)+","+str(args.xcropstart)+",1 --y-region="+str(args.ycropstart-args.ycroplength)+","+str(args.ycropstart)+",1 "

# phase retreival
energy = args.energy
distance = args.distance
pixelsize = args.pixelsize
deltabeta = args.deltabeta
regrate = 0.4339*np.log(deltabeta)+0.0034
if args.phasemode == 0:
  if args.ringremovalmode == 0:
    phasecommand = "--absorptivity"
  elif args.ringremovalmode == 1:
  	phasecommand = "--projection-filter none --absorptivity"
elif args.phasemode == 1:
  if args.ringremovalmode == 0:
    phasecommand = "--disable-projection-crop --delta 1e-6 --energy "+str(energy)+" --propagation-distance "+str(distance)+" --pixel-size "+str(pixelsize)+" --regularization-rate "+str(regrate)
  elif args.ringremovalmode == 1:
    phasecommand = "--projection-filter none --delta 1e-6 --energy "+str(energy)+" --propagation-distance "+str(distance)+" --pixel-size "+str(pixelsize)+" --regularization-rate "+str(regrate)

# ring removal sinogram padding
padwidth, padheight, padx, pady = util2.calc_padding(width, number)
sigmah = args.sigmah
sigmav = args.sigmav

################## COMMANDS ##################

# STEP 1 - create directory variables and check darks flats and tomo
print("check raw data directory")
check1 = os.path.isdir(os.path.join(PATH,flatsname))
check2 = os.path.isdir(os.path.join(PATH,darksname))
check3 = os.path.isdir(os.path.join(PATH,tomoname))
if [check1,check2,check3] == [True,True,True]:
	pass
else:
	print("Incomplete Data")
	quit()


# STEP 2 - Reconstruction
if region_thick == 1:
	outputcommand = SAVE
else:
	outputcommand = os.path.join(SAVE,"sli","sli")

# reconstruct without ring removal
if args.ringremovalmode == 0:
	os.system('tofu reco --overall-angle 180 '+'--projections '+os.path.join(PATH,tomoname)+' --flats '+os.path.join(PATH,flatsname)+' --flat-scale 1.0 --darks '+os.path.join(PATH,darksname)+' --dark-scale 1.0'+' --output '+outputcommand+' --fix-nan-and-inf '+phasecommand+' --center-position-x '+str(CoR)+' --volume-angle-z 0 --number '+str(number)+' '+regioncommand+' --output-bytes-per-file 0')
	os.system("rm -rf "+TEMP)

# reconstruct with ring removal
elif args.ringremovalmode == 1:
	os.system("tofu preprocess "+'--projections '+os.path.join(PATH,tomoname)+' --flats '+os.path.join(PATH,flatsname)+' --flat-scale 1.0 --darks '+os.path.join(PATH,darksname)+' --dark-scale 1.0'+" --output "+TEMP+"/proj-step1/proj-%04i.tif --fix-nan-and-inf "+phasecommand+" --output-bytes-per-file 0")
	os.system("tofu sinos --projections "+TEMP+"/proj-step1 --output "+TEMP+"/sinos/sin-%04i.tif --number "+str(number)+" --y "+str(region_start)+" --height "+str(region_thick)+" --y-step 1 --output-bytes-per-file 0")
	os.system('ufo-launch read path='+TEMP+'/sinos ! pad x='+str(padx)+' width='+str(padwidth)+' y='+str(pady)+' height='+str(padheight)+' addressing-mode=mirrored_repeat ! fft dimensions=2 ! filter-stripes horizontal-sigma='+str(sigmah)+' vertical-sigma='+str(sigmav)+' ! ifft dimensions=2 crop-width='+str(padwidth)+' crop-height='+str(padheight)+' ! crop x='+str(padx)+' width='+str(width)+' y='+str(pady)+' height='+str(number)+' ! write filename="'+TEMP+'/sinos-filt/sin-%04i.tif" bytes-per-file=0 tiff-bigtiff=False')
	os.system("tofu sinos --projections "+TEMP+"/sinos-filt --output "+TEMP+"/proj-step2/proj-%04i.tif --number "+str(region_thick)+" --output-bytes-per-file 0")
	os.system('tofu reco --overall-angle 180  --projections '+TEMP+'/proj-step2 --output '+outputcommand+' --center-position-x '+str(CoR)+' --volume-angle-z 0 --number '+str(number)+' '+regioncommand+' --output-bytes-per-file 0')
	os.system("rm -rf "+TEMP)


# record log
loglines = ["Relavant paths and names:" \
   ,"Raw data path = "+PATH \
   ,"Name of flats folder = "+flatsname \
   ,"Name of darks folder = "+darksname \
   ,"Name of tomo folder = "+tomoname \
   ,"Reconstruction path = "+SAVE \
   ,"Temporary folder path = "+TEMP \
   ,"" \
   ,"Projection values:" \
   ,"Number of projections = "+str(number) \
   ,"Projection height = "+str(height) \
   ,"Projection width = "+str(width) \
   ,"Center of rotation = "+str(CoR) \
   ,"" \
   ,"Reconstruction region:" \
   ,"First reconstructed slice = "+str(region_start) \
   ,"Total reconstructed slices = "+str(region_thick) \
   ,"cropmode = "+str(args.cropmode) \
   ,"xcropstart = "+str(args.xcropstart) \
   ,"xcroplength = "+str(args.xcroplength) \
   ,"ycropstart = "+str(args.ycropstart) \
   ,"ycroplength = "+str(args.ycroplength) \
   ,"" \
   ,"Phase retrieval:" \
   ,"Use phase retrieval ="+str(args.phasemode) \
   ,"X-ray energy = "+str(energy) \
   ,"Sample to detector distance = "+str(distance) \
   ,"Effective pixel size = "+str(pixelsize) \
   ,"delta/beta = "+str(deltabeta) \
   ,"" \
   ,"Ring removal:" \
   ,"Use ring removal ="+str(args.ringremovalmode) \
   ,"Horizontal sigma = "+str(sigmah) \
   ,"Vertical sigma = "+str(sigmav)]

# write log file
with open(os.path.join(SAVE,"reconstruction_log.txt"), 'w') as f:
	f.write('\n'.join(loglines))

# print time elapsed
elapsedtime = str(datetime.now()-starttime)
print("Finisehd in "+elapsedtime)
starttime = datetime.now()




