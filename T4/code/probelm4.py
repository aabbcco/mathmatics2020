# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:15:56 2020

@author: 426-2019级-1
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df = pd.read_excel("predict_mor.xlsx")

mor = df['MOR_PREDICT']

'''
画图观察是否为平稳序列

plt.figure(figsize=(10,6))
plt.plot(df.index,mor)
plt.show() #看上去不平稳
'''

'''
一阶差分
'''
def timestamp(h,m,s,gap,num):
    for i in range(num):
        s = s+1 if i%2 == 0 else s
        s+=gap
        m+=int(s/60)
        h+=int(m/60)
        s = s%60
        m = m%60
    return "2016-04-14 %s:%s:%s"%(
        str(h) if h>=10 else '0'+str(h),
        str(m) if m>=10 else '0'+str(m),
        str(s) if s>=10 else '0'+str(s),
        )
    

time = df['date']
mor_d1 = np.diff(mor)
#mor_d2 = np.diff(mor_d1)
#plt.plot(mor)
##plt.title("高速公路MOR估算时序图",fontsize=30)
#plt.xlabel("时间序列",fontsize=25)
#plt.ylabel("MOR(m)")
#plt.show() #一阶差分 大致稳定
'''
plt.figure()
plt.plot(range(len(mor_d1)),mor_d1)
plt.title("一阶差分",fontsize=30 )
plt.xlabel("时间序列",fontsize=25)
plt.ylabel("MOR一阶差分值",fontsize=25)
plt.figure()
plt.plot(range(len(mor_d2)),mor_d2)
plt.title("二阶差分",fontsize=30) 
plt.xlabel("时间序列",fontsize=25)
plt.ylabel("MOR二阶差分值",fontsize=25)
'''
#from statsmodels.tsa.stattools import adfuller
#adf = adfuller(mor)
#print(adf)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
'''
plot_acf(mor_d1)
plt.xlabel("p",fontsize=25)
plt.title("自相关图",fontsize=30)
plot_pacf(mor_d1)
plt.xlabel("q",fontsize=25)
plt.title("偏自相关",fontsize=30)
'''
'''
(-9.482240734386155,
 3.845143230413058e-16, 
 2, 95, {'1%': -3.5011373281819504, '5%': -2.8924800524857854, 
 '10%': -2.5832749307479226}, 522.7009913785289)
时序信号自身adf为-9.4822 均小于三种置信度 因此可以认作平稳信号
'''

#使用ARIMA去拟合原始数据，使用ARMA去拟合一阶差分数据 这里就使用ARMA模型
train = mor[0:80]
test = mor[80:-1]
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(train,order = (15,1,1)) #p,q来自于上面 d为几阶差分后可认作为平稳信号
result = model.fit()


'''
残差检验
resid = result.resid
from statsmodels.graphics.api import qqplot
qqplot(resid,line = 'q',fit = True)
plt.show()  #qq图上 红线为正态分布 即红线 结果可以看出散点图大致符合该趋势 因此信号为白噪声

'''

plt.figure()
pred = result.predict(start=1,end=len(mor)+200,typ='levels')
x=100
for i in range(100,len(pred)):
    if pred[i] >= 220:
        x = i
        break


plt.xticks([0,98],['公路截图开始时间\n'+time[0],'公路截图结束时间\n2016-04-14 07:39:11'])
plt.plot(range(len(pred)),[220]*len(pred),linestyle = '--')
plt.plot(range(len(mor)),mor,c='r')
plt.plot(range(len(pred)),pred,c='g')
plt.title('ARIMA模型预测MOR以及计算估计所得MOR',fontsize=30)
plt.annotate('预测大雾消散时间:\n%s'%timestamp(6,31,8,41,x), xy=(x, pred[x]), xycoords='data', xytext=(-100, -100),
             textcoords='offset points', fontsize=20,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2")
            )
sum_abs = 0
for i in range(79,99):
    sum_abs = abs(pred[i]-mor[i])/pred[i]
print(sum_abs/20)
plt.tick_params(labelsize=20)
plt.legend(['期望的mor','计算估计的mor','模型预测的mor'],fontsize=25)
plt.xlabel('时间序列',fontsize=25)
plt.ylabel('MOR(m)',fontsize=25)

plt.show()


















