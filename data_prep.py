import numpy as np
import pandas as pd

ptu1 = pd.read_excel("PTU_R06_12.xlsx")

vis1=pd.read_excel("VIS_R06_12.xlsx")

wind1 = pd.read_excel("WIND_R06_12.xlsx")


print('ptu',ptu1.columns)
print('vis',vis1.columns)
print('winds',wind1.columns)

#merge wind and vis
data = pd.DataFrame()
data["LOCALDATE (BEIJING)"] = vis1["LOCALDATE (BEIJING)"]
data["MOR_1A"] = vis1["MOR_1A"]
data["RVR_1A"] = vis1["RVR_1A"]
data["WS2A (MPS)"] = wind1["WS2A (MPS)"]
data["WD2A"] = wind1["WD2A"]
data["CW2A (MPS)"] = wind1["CW2A (MPS)"]


PAINS=[]
QFE=[]
QNH=[]
TEMP=[]
RH=[]
DEWPOINT=[]

ptu1_pointer = 0
data_pointer=0
while(1):
    if (data["LOCALDATE (BEIJING)"][data_pointer]).strftime('%H%M') == (ptu1["LOCALDATE (BEIJING)"][ptu1_pointer]).strftime('%H%M'):
        PAINS.append(ptu1["PAINS (HPA)"][ptu1_pointer])
        QFE.append(ptu1["QFE R06 (HPA)"][ptu1_pointer])
        QNH.append(ptu1["QNH AERODROME (HPA)"][ptu1_pointer])
        TEMP.append(ptu1["TEMP (°C)"][ptu1_pointer])
        RH.append(ptu1["RH (%)"][ptu1_pointer])
        DEWPOINT.append(ptu1["DEWPOINT (掳C)"][ptu1_pointer])
        data_pointer+=1
        if data_pointer == data["LOCALDATE (BEIJING)"].shape[0]:
            break
    else:
        ptu1_pointer+=1
        print(ptu1_pointer)

print(data["LOCALDATE (BEIJING)"].shape[0])

data["PAINS (HPA)"]=PAINS
data["QFE R06 (HPA)"]=QFE
data["QNH AERODROME(HPA)"]=QNH
data["TEMP (°C)"]=TEMP
data["RH"]=RH
data["DEWPOINT (掳C)"]=DEWPOINT

writer=pd.ExcelWriter('./duplicated.xlsx')
data.to_excel(writer)
writer.save()
writer.close()
print("done")
