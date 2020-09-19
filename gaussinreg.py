import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np 
from sklearn.utils import shuffle

X = pd.read_excel("combined.xlsx")
col = X.columns
print(col)
X1 = []
target = []
feature_name = []

X1.append(X[col[6]])
X1.append(X[col[7]])
feature_name.append(col[6])
X1.append(abs(X[col[7]]-X[col[9]]))
X1.append(X[col[8]])
# for i in range(13,len(col)):
#     X1.append(X[col[i]])
#     feature_name.append(col[i])
X1.append(X[col[13]])
X1.append(X[col[15]])
for i in range(10,12):
    target.append(X[col[i]])

target = np.asarray(target).transpose((1,0))
print(np.asarray(target).shape)
data = np.asarray(X1).transpose((1,0))


target = target[:,1]
target = np.asarray(target)
print(target.shape)
print(target)

data,target= shuffle(data,target)

train = data[300:,:]
test = data[:300,:]
train_target=target[300:]
test_target = target[:300]

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


plt.scatter(range(train_target.shape[0]),train_target,s=1)
plt.scatter(range(train_target.shape[0]),ff,color = 'darkorange',s=1)
plt.show()

plt.scatter(range(test_target.shape[0]),test_target,s=1)
plt.scatter(range(test_target.shape[0]),f, color = 'darkorange',s=1)
plt.xlabel('X')
plt.ylabel('y')
plt.show()
