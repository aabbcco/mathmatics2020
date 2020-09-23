import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd


loss = pd.read_csv("loss.csv")
acc = pd.read_csv("acc.csv")

plt.plot(loss['Step'],loss['Value'])
plt.xlabel('step')
plt.ylabel('loss')
plt.title('loss')
plt.show()
plt.plot(acc['Step'],acc["Value"])
plt.xlabel('step')
plt.ylabel('acc')
plt.title('acc')
plt.show()
