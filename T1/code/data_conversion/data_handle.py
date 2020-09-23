import time
import numpy as np
import pandas as pd 
'''
这段代码是用来进行第一问的数据预处理 功能为提取与构建模型相关联的数据
'''

df1 = pd.read_excel("PTU_R06_15.xlsx") #size正常
df2 = pd.read_excel("VIS_R06_15.xlsx") #不正常
df3 = pd.read_excel("WIND_R06_15.xlsx") #不正常
col1 = df1.columns
col2 = df2.columns
col3 = df3.columns
date = df2['CREATEDATE']
df2['minsecond'] =date.apply(lambda x:x.strftime('%M%S'))
count = 0
a=[]
for i in range(len(df2['minsecond'])-1): #
    if(df2['minsecond'][i] != df2['minsecond'][i+1]): 
        line = dict(df2.loc[i,:])  #满足整点时间，添加
        a.append(line)
a.append(dict(df2.loc[len(df2['minsecond'])-1,:]))
new_df2 = pd.DataFrame(a)
a=[]
date = df3['CREATEDATE']

df3['second'] =date.apply(lambda x:x.strftime('%S'))
for i in range(len(df3['second'])):
    if(df3['second'][i] == "00"):
        line = dict(df3.loc[i,:]) #满足整点时间，添加
        a.append(line)
new_df3 = pd.DataFrame(a)

new_df = pd.DataFrame(df1['CREATEDATE'])
new_df['LOCALDATE (BEIJING)'] = df1['LOCALDATE (BEIJING)']
new_df['SITE'] = df1['SITE']
new_df['PAINS(HPA)'] = df1['PAINS (HPA)']
new_df['QFE 06'] = df1['QFE R06 (HPA)']
new_df['QNH'] = df1['QNH AERODROME (HPA)']
new_df['TEMP'] = df1['TEMP (C)']
new_df['RH'] = df1['RH (%)']
new_df['DEWPOINT'] = df1['DEWPOINT (C)']
new_df['RVR_1A'] = new_df2['RVR_1A']
new_df['MOR_1A'] = new_df2['MOR_1A']
new_df['MOR_RAW'] = new_df2['MOR_RAW']
new_df['LIGHT_S'] = new_df2['LIGHTS']
new_df['WS2A'] = new_df3['WS2M (MPS)']
new_df['WD2A'] = new_df3['WD2A']
new_df['CW2A'] = new_df3['CW2A (MPS)']
new_df.to_excel('concat_data_15.xlsx')



'''
这段代码用于两个xlsx文件合并
'''
'''
df4 = pd.read_excel('concat_data_12.xlsx')
df5 = pd.read_excel('concat_data_15.xlsx')
result = df4.append(df5)
result.to_excel('combined.xlsx')
'''