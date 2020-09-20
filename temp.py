import pandas as pd
data = pd.read_excel('concat_data_15.xlsx')
import numpy as np
from matplotlib import pyplot as torch
base = np.asfarray(abs(data["DEWPOINT"]-data['TEMP'])).transpose()
target = np.asfarray(data['MOR_RAW'])
torch.scatter(base,target,s=1)
torch.xlabel('abs(DEWPOINT-TEMP)')
torch.ylabel('MOR')
torch.title('abs(DEWPOINT-TEMP)---MOR')
torch.savefig("absolute_temp 15.png")
torch.show()