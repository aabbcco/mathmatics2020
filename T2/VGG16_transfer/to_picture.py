import os
import numpy as np
import cv2

def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]

files = getFiles('results','.npy')

for _,name in enumerate(files):
    data =np.load(name)
    data = data.transpose(1,2,0)#CHW to HWC
    data*=255.0
    cv2.imwrite('results_picture/'+(((name.split('/'))[-1]).split('.'))[0]+'.png',data)

