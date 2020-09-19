import numpy as np
import pandas as pd 
import math
import os
import cv2

w = 100
h = 100
c = 3
def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist

def read_img(path):

    imgs=[]
    #label1=[]
    #label2 = []                                                                                                                                                                                                                                                                                                                                                                                                                                                    11111111
    img_name = get_filelist(path)
    for im in img_name:
            print('reading the images:%s'% (im))
            img = cv2.imread(im)
            imgs.append(img)
    return np.asarray(imgs, np.float64)

# df = pd.read_csv("label_2.csv")
# label1 = df['RVR']
# label2 = df['MOR']
# 样本和标签的读入与分类
path = "after_handle_2"
data = read_img(path)
print(np.asarray(data).shape)
label =np.loadtxt('label.txt')
label =label.astype(np.int)
label = label.reshape((len(label)))
#打乱顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
data = data.transpose((0,3,1,2))
print(data.shape)
label = label[arr]
print(label.shape)



#将所有数据分为训练集和验证集
ratio = 0.8
s = np.int(num_example*ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

np.save("x_train.npy",x_train)
np.save("y_train.npy",y_train)
np.save("x_val.npy",x_val)
np.save("y_val.npy",y_val)
