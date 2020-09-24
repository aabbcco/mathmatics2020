import cv2
import os
import pandas as pd

def tailor(img,length,x,y):
    return img[y:y+length,x:x+length] #行列

df = pd.read_excel("VIS_R06_12.xlsx")
#这一段是resize
'''
for root,dir,files in os.walk("C:/Users/Administrator/Desktop/clip"):
    for file in files:
        img = cv2.imread(root+'/'+file)
        img1 = cv2.resize(img,(100,100),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('C:/Users/Administrator/Desktop/resize/'+file,img1) 
'''
LOCALDATE = df['LOCALDATE (BEIJING)']
df['day'] =LOCALDATE.apply(lambda x:x.strftime('%d'))  #只取3/13的数据
df['second'] = LOCALDATE.apply(lambda x:x.strftime('%S')) #因为有的数据是15s一采集 有的20s一采集
a = []

for i in range(2,len(df['day'])):
    if df['day'][i] == '13':
        a.append(dict(df.loc[i,:])) #添加
new_df = pd.DataFrame(a) #还未处理完毕 需要在删去跳过的帧和起始帧
label_mor = []
label_rvr = []


for i in range(5,new_df.shape[0]):
    if i <= 47*4+2 or i >= 61*4+3: #视频图像48分到61:45分无图像 跳过
        if df['second'][i] == '40':
            label_mor.append(new_df['MOR_1A'][i-1])
            label_rvr.append(new_df['RVR_1A'][i-1])
        label_mor.append(new_df['MOR_1A'][i])
        label_rvr.append(new_df['RVR_1A'][i])

        #pains.append(new_df[''])
img_num = len(label_mor)
h = 720
w = 1280
imgs = 0
img_list = []
#裁剪 处理完放入tailor文件夹
for root,dir,files in os.walk("C:/Users/Administrator/Desktop/math/data/clip"):
    for i in range(img_num):
        img = cv2.imread(root+'/'+files[i])
        img = img[110:600,:] #无水印照片
        imgs+=1
        cv2.imwrite('C:/Users/Administrator/Desktop/math/data/non_mask/'+str(imgs)+'.png',img) #这里还不需要添加标签 因为label_mor label_rvr存的就是他们的标签数据
        dst = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('C:/Users/Administrator/Desktop/math/data/after_handle/'+str(imgs)+'.png',dst) #resize

print(pd.value_counts(label_mor))

'''

for i in range(k): #前k张为原始没有水印的图片
    if label_mor[i] != 50:
        img = cv2.imread('C:/Users/Administrator/Desktop/math/data/after_handle/'+str(i+1)+'.png')
        img1 = cv2.flip(img,1)
        img2 = cv2.flip(img,0)
        img3 = cv2.flip(img,-1)
        cv2.imwrite('C:/Users/Administrator/Desktop/math/data/after_handle/'+str(imgs+1)+'.png',img1)
        cv2.imwrite('C:/Users/Administrator/Desktop/math/data/after_handle/'+str(imgs+2)+'.png',img2)
        cv2.imwrite('C:/Users/Administrator/Desktop/math/data/after_handle/'+str(imgs+3)+'.png',img3)
        imgs+=3
        for j in range(3):
            label_mor.append(label_mor[i])
            label_rvr.append(label_rvr[i])
'''

k = imgs
label_file = open('label.csv','w')
for i in range(k):
    label_file.write('%s,%s\n'%(label_mor[i],label_rvr[i]))
label_file.close()

#label_df = new_df['MOR_1A']
#print(label_df)