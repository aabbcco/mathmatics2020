import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np 
from sklearn.utils import shuffle

data = np.load('data.npy')
target = np.load('target.npy')

feature_name = ['QNH','TEMP','RH','absolute_temp','WS2A','CW2A']

train = data[200:,:]
test = data[:200,:]
train_target=target[200:]
test_target = target[:200]

gpr =GaussianProcessRegressor(optimizer='fmin_l_bfgs_b')
gpf=gpr.fit(train,train_target)
f=gpr.predict(test)
ff=gpr.predict(train)

print(gpf)

truth=0
for i in range(test_target.shape[0]):
    if test_target[i]==0 and abs(f[i])<10:
        truth+=1
        continue
    if abs(test_target[i]-f[i])/test_target[i]<0.1:
        truth+=1

print("truth: ",truth/test_target.shape[0])

para=gpr.kernel_.theta

print(para)

from matplotlib import pyplot as plt


plt.scatter(range(train_target.shape[0]),train_target,label='label',s =2)
plt.scatter(range(train_target.shape[0]),ff,label='pred',s=2)
plt.title("Gaussin-Train")
plt.legend(loc="upper right")
plt.show()

plt.scatter(range(test.shape[0]), test_target,label="label",s=2)
plt.scatter(range(test.shape[0]), f, color='darkorange', label='pred',s=2)
plt.title('Gaussin')
plt.legend(loc="upper right")
plt.xlabel('X')
plt.ylabel('y')
plt.show()
