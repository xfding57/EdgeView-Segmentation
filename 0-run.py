import os
import numpy as np
import subprocess
run = 1


samples = ["/staff/dingx/Desktop/Local_data/dingx/221217-prj36G12594-Hydrogels/rec/2023-05-30-ID/XFDing/Xiaoman/NSI_002_L-2splits-rec-train-prn2i/set1"]

for i in range(len(samples)):
	PATH = samples[i]
	SAVE = os.path.split(samples[i])[0]
	os.system("python 2-train.py -PATH "+PATH+" -SAVE "+SAVE+'-train -lossfunc "ssim 1" -bitrate 16 -epochs 20000 -epochsave 100')



