import pandas as pd


label = "./data_prep/label_224x224.csv"
df = pd.read_csv(label)
mors = df['MOR']

num_dict = {}

for i, mor in enumerate(mors):
    if not str(mor) in num_dict:
        num_dict[str(mor)] = 1
    else:
        num_dict[str(mor)] = num_dict[str(mor)] + 1



print(num_dict)
print(mors.shape[0])
print(1446/1857)