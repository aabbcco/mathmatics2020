import numpy as np
import pandas as pd
from sklearn import preprocessing

from rulefit import RuleFit



X = pd.read_excel("combined.xlsx")
col = X.columns
print(col)
X1 = []
target = []
feature_name = []

for i in range(6,9):
    X1.append(X[col[i]])
    feature_name.append(col[i])
X1.append(abs(X[col[7]]-X[col[9]]))
feature_name.append('absolute_temp')
for i in range(13,14):
    X1.append(X[col[i]])
    feature_name.append(col[i])

feature_name.append(col[15])
X1.append(X[col[15]])
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

from sklearn.utils import shuffle
data,target= shuffle(data,target)

train = data[200:,:]
test = data[:200,:]
train_target=target[200:]
test_target = target[:200]

from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.01)


relu_fit = RuleFit()
relu_fit.max_iter=4000
relu_fit.tree_generator = gb
relu_fit.fit(train,train_target,feature_names=feature_name)
f=relu_fit.predict(test)
ff = relu_fit.predict(train)
rule = relu_fit.get_rules()
truth=0
for i in range(test_target.shape[0]):
    if abs(test_target[i]-f[i])/test_target[i]<0.1:
        truth+=1

print("truth: ",truth/test_target.shape[0])
#print(rule)
ruleset = pd.DataFrame(data=rule)
writer=pd.ExcelWriter('./rules.xlsx')
ruleset.to_excel(writer)
writer.save()
writer.close()


from matplotlib import pyplot as plt



plt.scatter(range(train_target.shape[0]),train_target)
plt.scatter(range(train_target.shape[0]),ff)
plt.show()

plt.scatter(range(test_target.shape[0]),test_target)
plt.scatter(range(test_target.shape[0]),f, color = 'darkorange')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
