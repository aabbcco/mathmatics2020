import cv2
import numpy as np 
from dark import dehaze as dcp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttf')
plt.rcParams['font.sans-serif']='SimHei' #中文字体设置

pre = 'C:/Users/426-2019级-1/Desktop/数模题目/2020年E题/高速公路视频截图/'
def len2num(a):
    if a < 10:
        return '0' + str(a)
    else:
        return str(a)

def timestamp(h,m,s,gap,num): #开始的小时 分钟 秒 以及采样时间间隔(s) , 采样次数
    stamp = []
    for i in range(num):  
        s = s + 1 if i%2 == 0 else s
        stamp.append('2016-04-14 '+len2num(h)+':'+len2num(m)+':'+len2num(s))
        s = s + gap
        m = m + int( s / 60)
        h = h + int( m / 60)
        s = s % 60
        m = m % 60
    return stamp

def cv_imread(path):
    return cv2.imdecode(np.fromfile(path,np.uint8), cv2.IMREAD_UNCHANGED)

def cv_imwrite(path,frame):
    cv2.imencode('.png', frame)[1].tofile(path)  #cv2 用gbk np用utf-8 不兼容只能这么写

def camera_calibration(img):
    mask = cv_imread('高速公路ROI.png') 
    #kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #sobel算子
    dst = cv2.Canny(img,20,52) #需要调参的地方 canny边缘提取算子
    margin = cv2.bitwise_and(dst,mask) #滤除文字水印
    h,w = margin.shape
    edges = np.zeros((h,w),dtype=np.uint8)

    cv2.threshold(margin,100,255,cv2.THRESH_BINARY,dst = edges)#阈值过滤
    index = np.where(edges>150) #找出边缘的坐标
    index = np.transpose(index) 
    for xy in index: #滤除不需要的坐标
        x,y = xy
        if y<390 or y>800 or x<500: #提取感兴趣区域
            edges[x,y] = 0

    lines = cv2.HoughLines(edges,1,np.pi/120,115) #需要调参，霍夫曼法画直线
    
    k_list = [] #斜率
    b_list = [] #截距
    new_edges = np.zeros((h,w,3),dtype = np.uint8)#用来存放灭点以及展示图像

    for line in lines:
        for rho,theta in line: 
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1500*(-b))
            y1 = int(y0 + 1500*(a))
            x2 = int(x0 - 1500*(-b))
            y2 = int(y0 - 1500*(a))
            cv2.line(new_edges,(x1,y1),(x2,y2),(0,0,255),1)
    cv2.imshow("dst",new_edges)
    cv2.waitKey()
    cv2.destroyAllWindows()
    ''' 计算灭点方法--求解直线方程交点
            k = (y1-y2) / (x1-x2)
            b = (y1-k*x1)
            k_list.append(k)
            b_list.append(b)
    k1,k2 = k_list
    b1,b2 = b_list
    x_cross = int((b2-b1)/(k1-k2))
    y_cross = int(k1*x_cross + b1)
    cv2.circle(new_edges,(x_cross,y_cross),2,(255,255,0),1) #标出灭点
    '''
    #灭点坐标通过PS手动标定坐标为(300,291) 论文中可以通过直线方程求得
    dead_point = (300,291)
    triangle = np.array([[300,291],[478,719],[801,719]]) #车道线的三个坐标 PS标定
    lane_ROI=cv2.fillConvexPoly(np.zeros((h,w),dtype=np.uint8),triangle,255)#车道亮度ROI
    cv2.circle(new_edges,dead_point,2,(0,255,255),1) #标出灭点


    _,labels,stats,centres = cv2.connectedComponentsWithStats(edges) #labels标签图 stats:x0 y0左上角boundingbox坐标 width heigth boundingbox的尺寸 area面积
    boundingbox = edges.copy()
    boundingboxwithoutlapping = boundingbox.copy()

    bbi=[] #bounding box index
    for i in range(1,len(stats)):#删去重叠的boundingbox index从1开始 因为0是背景
        cv2.rectangle(boundingbox,(stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]),128)
        j = 1
        flag = 0 #是否画boundingbox 0:画 1不画 
        for j in range(1,len(stats)):
            if(j != i):  #防止同一个boundingbox
                x0,y0,w0,h0,area0 = stats[i] 
                x1,y1,w1,h1,area1 = stats[j]
                x0_set = set(range(x0,x0+w0))
                y0_set = set(range(y0,y0+h0))
                x1_set = set(range(x1,x1+w1))
                y1_set = set(range(y1,y1+h1))
                overlapx = x0_set&x1_set
                overlapy = y0_set&y1_set
                if overlapx and overlapy:#重叠
                    flag = 1
                    break 
            else: #不重叠 判断下一个
                continue
        if(flag == 0): #不和其他所有boundingbox重叠
            bbi.append(stats[i])
            cv2.rectangle(boundingboxwithoutlapping,(stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]),128)
    v2 = bbi[0][0]+bbi[0][2] #相当于头端
    v1 = bbi[1][0]  #相当于尾端
    vh = dead_point[0]
    f = 9/(1/(v2-vh)-1/(v1-vh)) #因为用的是两根车道分界线的间距 为9m， f为相机参数
    return f,vh,lane_ROI

'''
以上为标定过程
'''
#用于标定的帧
img_for_calibration = cv_imread(pre+"original_frame1.bmp")
f,vh,roi= camera_calibration(img_for_calibration)
MOR_list = []
time_list = timestamp(6,31,8,41,99)
for frame in range(99):
    img = cv_imread(pre+'original_frame%s.bmp'%(frame+2))
    b = img[:,:,0].astype(np.float)
    g = img[:,:,1].astype(np.float)
    r = img[:,:,2].astype(np.float)
    I = 0.5*(np.maximum(np.maximum(b,g),r)+np.minimum(np.minimum(b,g),r)) #HSL模型亮度定义
    h,w = I.shape
    C = np.zeros((h,w),dtype = np.float)
    xy = [] #满足条件的xy坐标点
    
    '''
    亮度对比度四领域内计算
    '''
    up = np.zeros((h,w))
    down = np.zeros((h,w))
    left = np.zeros((h,w))
    right = np.zeros((h,w))

    for i in range(1,h-1):
        up[i,:] = np.abs(I[i,:]-I[i-1,:])/np.maximum(I[i,:],I[i-1,:])
        down[i,:] = np.abs(I[i,:]-I[i+1,:])/np.maximum(I[i,:],I[i+1,:])
    for j in range(1,w-1):
        left[:,j] = np.abs(I[:,j]-I[:,j-1])/np.maximum(I[:,j],I[:,j-1])
        right[:,j] = np.abs(I[:,j]-I[:,j+1])/np.maximum(I[:,j],I[:,j+1])
    up[np.isnan(up)] = 0 #填充异常值
    down[np.isnan(down)] = 0
    left[np.isnan(left)] = 0
    right[np.isnan(right)] = 0
    C = np.maximum(up,down) #取最大
    C = np.maximum(C,left)
    C = np.maximum(C,right)
    
    
    calculate_lane_ROI = roi.astype(np.float) #ROI转换到浮点数进行计算
    calculate_lane_ROI /= 255
    C = C * calculate_lane_ROI #相当于在原来lane_ROI中非感兴趣区域置零

    farest_x = 718 #最差情况
    for i in range(vh+1,h):
        if np.count_nonzero(C[i,:]>=0.05) > 10: #对比度大于0.05的行像素点数大于10个 作为最远可视
            farest_x = i
            break
    
    t,recover = dcp.dehaze(img)
    t = t*calculate_lane_ROI #相当于在原来lane_ROI中非感兴趣区域置零
    y_t = np.where(t[farest_x,:]!=0)
    mint = np.min(t[farest_x,y_t])
    depth = f / (farest_x - vh)
    k = - np.log(mint) / depth #mor = 3 / k
    MOR_list.append(int(3/k))
    print('img%s is ok'%(frame+2))
    
df = pd.DataFrame({'date':time_list,'MOR_PREDICT':MOR_list})
df.to_excel('predict_mor.xlsx',index=False)
plt.title("高速公路能见度变化趋势图",fontsize=30)
x = range(len(time_list))
plt.plot(x, MOR_list, 'ro-',markersize=8)
plt.xticks(x, time_list, rotation=90,fontsize=10)
plt.xlabel('时间',fontsize=20)
plt.ylabel('光学能见度MOR(m)',fontsize=20)
for a, b in zip(x, MOR_list):
    plt.text(a, b, b, ha='left', va='baseline', fontsize=15)

plt.show()
#print(MOR_list)

#可视化部分
'''
cv2.imshow("original",img)

cv2.imshow("lane and dead point",new_edges)
cv2.imshow("lapping box",boundingbox)
cv2.imshow("non-lapping box",boundingboxwithoutlapping)
cv2.waitKey()
cv2.destroyAllWindows()
'''
