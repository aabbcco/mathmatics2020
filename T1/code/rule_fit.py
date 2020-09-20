import numpy as np
import pandas as pd
from sklearn import preprocessing

from rulefit import RuleFit


data = np.load('data.npy')
target = np.load('target.npy')

feature_name = ['QNH','TEMP','RH','absolute_temp','WS2A','CW2A']

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

plt.scatter(range(train_target.shape[0]),train_target,label='label',s =2)
plt.scatter(range(train_target.shape[0]),ff,label='pred',s=2)
plt.title("Rulefit-Train")
plt.legend(loc="upper right")
plt.show()

plt.scatter(range(test.shape[0]), test_target,label="label",s=2)
plt.scatter(range(test.shape[0]), f, color='darkorange', label='pred',s=2)
plt.title('Rulefit')
plt.legend(loc="upper right")
plt.xlabel('X')
plt.ylabel('y')
plt.show()