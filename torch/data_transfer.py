import pandas as pd
import numpy as np
import os
from shutil import copyfile


datafolder = 'after_handle_224x224'
label = "label_224x224.csv"
dest_folder = '2020_2'

for i in range(5):
    os.mkdir(os.path.join('./', dest_folder, str(int(i))))

df = pd.read_csv(label)
label1 = df['MOR']

guide = np.zeros(label1.shape[0],dtype = np.int)

for i in range(label1.shape[0]):
    if label1[i] == 0:
        continue
    elif label1[i] == 50:
        copyfile(os.path.join('./',datafolder, str(i + 1) + '.png'), os.path.join(dest_folder, '0',str(i + 1) + '.png'))
    elif label1[i] == 100:
        copyfile(os.path.join('./',datafolder, str(i + 1) + '.png'), os.path.join(dest_folder, '1',str(i + 1) + '.png'))
    elif label1[i] <= 300:
        copyfile(os.path.join('./',datafolder, str(i + 1) + '.png'), os.path.join(dest_folder, '2',str(i + 1) + '.png'))
    elif label1[i] <= 700:
        copyfile(os.path.join('./',datafolder, str(i + 1) + '.png'), os.path.join(dest_folder, '3',str(i + 1) + '.png'))
    elif label1[i] <= 1200:
        copyfile(os.path.join('./',datafolder, str(i + 1) + '.png'), os.path.join(dest_folder, '4',str(i + 1) + '.png'))
    print(i)

        

