import cv2
import os
import pandas as pd

def tailor(img,length,x,y):
    return img[y:y+length,x:x+length] #行列

df = pd.read_excel("combined.xlsx")
#这一段是resize
'''
for root,dir,files in os.walk("C:/Users/Administrator/Desktop/clip"):
    for file in files:
        img = cv2.imread(root+'/'+file)
        img1 = cv2.resize(img,(100,100),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('C:/Users/Administrator/Desktop/resize/'+file,img1) 
'''
df = df.loc[0:df.shape[0]/2-1,:]
LOCALDATE = df['LOCALDATE (BEIJING)']
df['day'] =LOCALDATE.apply(lambda x:x.strftime('%d'))  #只取3/13的数据
a = []
for i in range(2,len(df['day'])):
    if df['day'][i] == '13':
        a.append(dict(df.loc[i,:])) #添加
new_df = pd.DataFrame(a) #还未处理完毕 需要在删去跳过的帧和起始帧
label_mor = []
label_rvr = []
for i in range(2,new_df.shape[0]):
    if i < 48 or i > 61: #视频图像48分到72分无图像 跳过
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
        dst = cv2.resize(img,(100,100),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('C:/Users/Administrator/Desktop/math/data/after_handle/'+str(imgs)+'.png',dst) #resize

k = imgs
for i in range(k): #前k张为原始没有水印的图片
    x0=0
    y0=0
    img = cv2.imread('C:/Users/Administrator/Desktop/math/data/non_mask/'+str(i+1)+'.png')
    for j in range(50-pd.value_counts(label_mor)[label_mor[i]]):
        if x0 > w-100 : #裁剪超出范畴
            x0 = 0
            y0 += 20
        img_tailored = tailor(img,100,x0,y0)
        x0+=30
        imgs+=1
        cv2.imwrite('C:/Users/Administrator/Desktop/math/data/after_handle/'+str(imgs)+'.png',img_tailored) #tailor并且只用标签少的数据
        label_mor.append(label_mor[i])
        label_rvr.append(label_rvr[i])
k = imgs
print(len(label_mor))
label_file = open('label.csv','w')
for i in range(len(label_mor)):
    label_file.write('%s,%s\n'%(label_mor[i],label_rvr[i]))
label_file.close()

#label_df = new_df['MOR_1A']
#print(label_df)
