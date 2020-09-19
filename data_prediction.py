import pandas as tf
import numpy as np
from matplotlib import pyplot as torch

data = tf.read_excel("concat_data_15.xlsx")

wanted = ['PAINS(HPA)','QFE 06','QNH','TEMP','RH','DEWPOINT','WS2A','WD2A','CW2A']

target = np.asfarray(data['MOR_1A'])

for i,key in enumerate(wanted):
    base = np.asfarray(data[key]).transpose()
    torch.scatter(base,target,s=1)
    torch.xlabel(key)
    torch.ylabel('neng jian du')
    torch.savefig(key+"15.png")
    torch.show()

