import numpy as np
import pandas as pd
from sklearn import preprocessing

from rulefit import RuleFit


from sklearn.utils import shuffle

X = pd.read_excel("combined.xlsx")
col = X.columns
print(col)
X1 = []
target = []
feature_name = []

for i in range(4,10):
    X1.append(X[col[i]])
    feature_name.append(col[i])
for i in range(12,len(col)):
    X1.append(X[col[i]])
    feature_name.append(col[i])
for i in range(10,12):
    target.append(X[col[i]])
    target = np.asarray(target).transpose((1,0))
print(np.asarray(target).shape)

#X1 = np.asarray(X1).transpose((1,0))
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(X1)
data = np.asarray(X1).transpose((1,0))


target = target[:,1]
target = np.asarray(target)
print(target.shape)
print(target)


xx = X1.as_matrix()

relu_fit = RuleFit()
relu_fit.fit(xx,target,feature_names=feature_name)
relu_fit.predict(xx)

