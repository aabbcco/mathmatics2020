import numpy as np
import pandas as pd

#{'1200': 2, '1300': 4, '1400': 2, '1500': 7, '1600': 5, '1100': 4, '1000': 13, '900': 25, '800': 14, '700': 2, '600': 3, '500': 9, '450': 11, '400': 12, '350': 12, '300': 27, '250': 25, '200': 42, '150': 47, '100': 129, '50': 1446, '0': 16} 

data =pd.read_csv('label_224x224.csv')
data = data['MOR']

dicts = {}

for i,mor in enumerate(data):
    if str(mor) not in dicts:
        dicts[str(mor)] = 1
    else:
        dicts[str(mor)]=dicts[str(mor)]+1

print(dicts)