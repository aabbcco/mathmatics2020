from sklearn.utils import shuffle
import pandas as pd
import numpy as np
X = pd.read_excel("combined.xlsx")
col = X.columns
print(col)
X1 = []
target = []


for i in range(6,9):
    X1.append(X[col[i]])
X1.append(abs(X['TEMP'] - X['DEWPOINT']))
X1.append(X['WS2A'])
X1.append(X['CW2A'])
for i in range(10,12):
    target.append(X[col[i]])

target = np.asarray(target).transpose((1,0))
print(np.asarray(target).shape)

#X1 = np.asarray(X1).transpose((1,0))
data = np.asarray(X1).transpose((1,0))


target = target[:,1]
target = np.asarray(target)
data,target= shuffle(data,target)

np.save('data.npy',data)
np.save('target.npy',target)