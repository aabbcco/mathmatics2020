'''
这段代码是用来进行第一问的数据预处理 功能为提取与构建模型相关联的数据
'''

import pandas as pd
import numpy as np
import cv2
df1 = pd.read_excel("PTU_R06_15.xlsx",encoding='gbk') # PTU表 需要PAINS(HPA) QFE 06 QNH TEMP RH DEWPOINT
df2 = pd.read_excel("VIS_R06_15.xlsx",encoding='gbk') # VIS表   需要RVR_1A MOR_1A LIGHT_S
df3 = pd.read_excel("WIND_R06_15.xlsx",encoding='gbk') # WIND表 需要WS2A WD2A CW2A 
new_df = pd.DataFrame(df1,columns = ['CREATEDATE','LOCALDATE (BEIJING)','SITE'])
new_df['PAINS(HPA)']=df1['PAINS (HPA)']
new_df['QFE 06']=df1['QFE R06 (HPA)']
new_df['QNH']=df1['QNH AERODROME (HPA)']
new_df['TEMP']=df1['TEMP (掳C)']
new_df['RH']=df1['RH (%)']
new_df['DEWPOINT']=df1['DEWPOINT (掳C)']
new_df['RVR_1A'] = df2['RVR_1A']
new_df['MOR_1A'] = df2['MOR_1A']
new_df['LIGHT_S'] = df2['LIGHTS']
new_df['WS2A'] = df3['WS2A (MPS)']
new_df['WD2A'] = df3['WD2A']
new_df['CW2A'] = df3['CW2A (MPS)']
new_df.to_excel('concat_data.xlsx')