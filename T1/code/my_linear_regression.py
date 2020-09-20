#第一问 能见度与地面气象观测数据之间的关系
#线性回归
import math
from sklearn import preprocessing
import numpy as np
import pandas as pd
#主体部分
def linear_loss(X,y,w,b):
    num_train = X.shape[0]  #训练的样本数 m
    num_feature = X.shape[1] #每个样本的特征数 n
    #预测值
    y_hat = np.dot(X, w) + b
    #损失函数
    loss = np.sum((y_hat - y)**2) / num_train
    #loss = np.sum((y_hat - y) ** 2)
    #参数的偏导 L= 0.5 * (w*x + b - y)^2 分别对 w,b 求导
    cha = []
    for i in range(0,num_train):
        cha.append(y_hat[i] - y[i])
    cha = np.asarray(cha)
    #print(cha.shape)
    dw = np.dot(X.T, (cha)) / num_train #行向量和列向量点乘结果 一个数

    db = np.sum((y_hat - y)) / num_train #一个数
    return y_hat, loss, dw, db

#参数初始化
def initialize_params(dims):
    w = np.random.rand(dims, 1)
    b = 0
    return w, b

#基于梯度下降的模型训练过程
def linear_train(X, y, learning_rate, epochs):
    w, b = initialize_params(X.shape[1])
    loss_list = []
    for i in range(1, epochs):
        #计算当前预测值，损失和参数偏导
        y_hat, loss, dw, db = linear_loss(X, y, w, b)
        loss_list.append(loss)
        # 基于梯度下降的参数更新过程
        w += -learning_rate * dw
        b += -learning_rate * db

        #打印迭代次数和损失
        if i % 10000 == 0:
            print('epoch %d loss %f' %(i, loss))

        #保存参数
        params = {
            'w': w,
            'b': b
        }

        #保存梯度
        grads = {
            'dw':dw,
            'db':db
        }

    return loss_list, loss, params, grads


from sklearn.decomposition import PCA

datas = np.load('data.npy')
targets = np.load('target.npy')

train = datas[200:,:]
test = datas[:200,:]
train_target=targets[200:]
test_target = targets[:200]

# pca = PCA(n_components=3)   #降到2维
# pca.fit(data)                  #训练
# newdata=pca.fit_transform(data)   #降维后的数据
# #PCA(copy=True, n_components=2, whiten=False)
# print(pca.explained_variance_ratio_)  #输出贡献率

# print("new_data:", newdata.shape)

# data = newdata
#标签log
"""
for i in range(0,len(target)):
    if target[i] != 0:
        target[i] = math.log(target[i],10)
print(target[0])
"""


loss_list, loss, params, grads = linear_train(train, train_target, 0.000001, 500)
print("参数：", params)

def predict(X, params):
    w = params['w']
    b = params['b']
    y_pred = np.dot(X, w) + b
    return y_pred

import matplotlib.pyplot as plt

y_pred = predict(test, params) #得到预测值


#训练过程的损失绘制
plt.plot(loss_list, color = 'blue')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

truth=0
for i in range(test_target.shape[0]):
    if abs(test_target[i]-y_pred[i])/test_target[i]<0.1:
        truth+=1

print("truth: ",truth/test_target.shape[0])

#预测值与真实值
plt.scatter(range(test.shape[0]), test_target,label="label",s=2)
plt.scatter(range(test.shape[0]), y_pred, color='darkorange', label='pred',s=2)
plt.title('Linear Regression')
plt.legend(loc="upper right")
plt.xlabel('X')
plt.ylabel('y')
plt.show()



