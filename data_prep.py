import numpy as np
import pandas as pd

ptu1 = pd.read_excel("PTU_R06_12.xlsx")
ptu2 = pd.read_excel("PTU_R06_15.xlsx")

vis1=pd.read_excel("VIS_R06_12.xlsx")
vis2=pd.read_excel("VIS_R06_15.xlsx")

wind1 = pd.read_excel("WIND_R06_12.xlsx")
wind2 = pd.read_excel("WIND_R06_15.xlsx")

print('ptu',ptu1.columns)
print('vis',vis1.columns)
print('winds',wind1.columns)

#merge wind and vis
data = pd.DataFrame()
data["LOCALDATE (BEIJING)"] = vis1["LOCALDATE (BEIJING)"]
data["LOCALDATE (BEIJING)"].append(vis2["LOCALDATE (BEIJING)"])
data["MOR_1A"] = vis1["MOR_1A"]
data["MOR_1A"].append(vis2["MOR_1A"])
data["RVR_1A"] = vis1["RVR_1A"]
data["RVR_1A"].append(vis2["RVR_1A"])
data["WS2A (MPS)"] = wind1["WS2A (MPS)"]
data["WS2A (MPS)"].append(wind2["WS2A (MPS)"])
data["WD2A"] = wind1["WD2A"]
data["WD2A"].append(wind2["WD2A"])
data["CW2A (MPS)"] = wind1["CW2A (MPS)"]
data["CW2A (MPS)"].append(wind2["CW2A (MPS)"])

#merge ptu
ptu1.append(ptu2)

ptu1_pointer = 0
for i in range(data["LOCALDATE (BEIJING)"].shape[0]):
    if (data["LOCALDATE (BEIJING)"][i]).strftime('%H%M') == (ptu1["LOCALDATE (BEIJING)"][ptu1_pointer]).strftime('%H%M'):
        data["PAINS (HPA)"].append(ptu1["PAINS (HPA)"][ptu1_pointer])
        data["QFE R06 (HPA)"].append(ptu1[ptu1["QFE R06 (HPA)"][ptu1_pointer]])
        data["QNH AERODROME(HPA)"].append(ptu1[ptu1["QNH AERODROME(HPA)"][ptu1_pointer]])
        data["TEMP (°C)"].append(ptu1[ptu1["TEMP (°C)"][ptu1_pointer]])
        data["RH (%)"].append(ptu1[ptu1["RH (%)"][ptu1_pointer]])
        data["DEWPOINT (掳C)"].append(ptu1[ptu1["DEWPOINT (掳C)"][ptu1_pointer]])
    else:
        ptu1_pointer+=1
        print(ptu1_pointer)

writer=pd.ExcelWriter('duplicated score')
data.to_excel(writer)
print("dones")
