#第一问 能见度与地面气象观测数据之间的关系
#线性回归

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
    b = 100
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
        w -= learning_rate * dw
        b -= learning_rate * db

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


from sklearn.utils import shuffle

X = pd.read_excel("concat_data20191216.xlsx")
col = X.columns
print(col)
X1 = []
target = []

for i in range(4,10):
    X1.append(X[col[i]])
for i in range(12,len(col)):
    X1.append(X[col[i]])
for i in range(10,12):
    target.append(X[col[i]])



#X1 = np.asarray(X1).transpose((1,0))
#print(np.asarray(X1).shape)
target = np.asarray(target).transpose((1,0))
print(np.asarray(target).shape)

#X1 = np.asarray(X1).transpose((1,0))
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(X1)
data = np.asarray(data).transpose((1,0))


target = target[:,1]
target = np.asarray(target)
print(target.shape)
print(target)

data, target = shuffle(data, target)
print(data[0])


#coding=utf-8

import numpy as np

from sklearn.decomposition import PCA


# # pca = PCA(n_components=4)   #降到2维
# pca.fit(data)                  #训练
# newdata=pca.fit_transform(data)   #降维后的数据
# # PCA(copy=True, n_components=2, whiten=False)
# print(pca.explained_variance_ratio_)  #输出贡献率

# print("new_data:", newdata.shape)

# data = newdata




loss_list, loss, params, grads = linear_train(data, target, 0.001, 10)
print("参数：", params)

def predict(X, params):
    w = params['w']
    b = params['b']
    y_pred = np.dot(X, w) + b
    return y_pred

import matplotlib.pyplot as plt

y_pred = predict(data, params) #得到预测值

import matplotlib.pyplot as plt

#训练过程的损失绘制
plt.plot(loss_list, color = 'blue')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#预测值与真实值
f = data.dot(params['w']) + params['b']
plt.scatter(range(data.shape[0]), target)
plt.plot(f, color = 'darkorange')
plt.xlabel('X')
plt.ylabel('y')
plt.show()



