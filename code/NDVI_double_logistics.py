# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:09:51 2022

@author: WH
"""

import pandas as pd
import numpy as np
import lmfit
import matplotlib.pyplot  as plt


# preprocess NDVI data
vi19 = pd.read_excel(r"G:\urban microclimate\data\India\india\delhi\NDVI\sentinel\2019.xlsx")
vi20 = pd.read_excel(r"G:\urban microclimate\data\India\india\delhi\NDVI\sentinel\2020.xlsx")
vi21 = pd.read_excel(r"G:\urban microclimate\data\India\india\delhi\NDVI\sentinel\2021.xlsx")
vi22 = pd.read_excel(r"G:\urban microclimate\data\India\india\delhi\NDVI\sentinel\2022.xlsx")

vi19.replace({-9999:np.nan},inplace=True)
vi20.replace({-9999:np.nan},inplace=True)
vi21.replace({-9999:np.nan},inplace=True)
vi22.replace({-9999:np.nan},inplace=True)

tree_ID = pd.read_csv(r"G:\urban microclimate\data\India\india\delhi\delhi_tree_main(ESA).csv")['ID']
vi19.index = vi19['Id']
vi20.index = vi20['Id']
vi21.index = vi21['Id']
vi22.index = vi22['Id']

d19 = vi19.columns.values.tolist()[1:]
d20 = vi20.columns.values.tolist()[1:]
d21 = vi21.columns.values.tolist()[1:]
d22 = vi22.columns.values.tolist()[1:]

vi_mean = pd.DataFrame(np.nan,columns=[x for x in range(366)],index=['19','20','21','22'],dtype='float')
for d in range(365): 
    if d in d19:
        lst_tree = []
        for i in vi19.index:
            if i in np.array(tree_ID):
                lst_tree.append(vi19[d][i])
            vi_mean[d]['19'] = np.nanmean(lst_tree)
    if d in d20:
        lst_tree = []
        for i in vi20.index:
            if i in np.array(tree_ID):
                lst_tree.append(vi20[d][i])
            vi_mean[d]['20'] = np.nanmean(lst_tree)
            
    if d in d21:
        lst_tree = []
        for i in vi21.index:
            if i in np.array(tree_ID):
                lst_tree.append(vi21[d][i])
            vi_mean[d]['21'] = np.nanmean(lst_tree)
            
    if d in d22:
        lst_tree = []
        for i in vi22.index:
            if i in np.array(tree_ID):
                lst_tree.append(vi22[d][i])
            vi_mean[d]['22'] = np.nanmean(lst_tree)
            
x1 = np.array(vi_mean.loc['19'].dropna().index)
y1 = np.array(vi_mean.loc['19'].dropna())
x2 = np.array(vi_mean.loc['20'].dropna().index)
y2 = np.array(vi_mean.loc['20'].dropna())
x3 = np.array(vi_mean.loc['21'].dropna().index)
y3 = np.array(vi_mean.loc['21'].dropna())
x4 = np.array(vi_mean.loc['22'].dropna().index)
y4 = np.array(vi_mean.loc['22'].dropna())



# smoothing temporally
doys = [x for x in range (16,344)]

vi_s = pd.DataFrame(np.nan,columns=doys,index = ['19','20','21','22'],dtype='float')

for i in doys:
    doy = [i+w for w in range(-15,16)]
    for y in vi_s.index:
        if np.isnan(vi_mean[i][y]) == False:
            vi_s[i][y] = np.nanmean(vi_mean.loc[y][doy])


x1 = np.array(vi_s.loc['19'].dropna().index)
y1 = np.array(vi_s.loc['19'].dropna())
x2 = np.array(vi_s.loc['20'].dropna().index)
y2 = np.array(vi_s.loc['20'].dropna())
x3 = np.array(vi_s.loc['21'].dropna().index)
y3 = np.array(vi_s.loc['21'].dropna())
x4 = np.array(vi_s.loc['22'].dropna().index)
y4 = np.array(vi_s.loc['22'].dropna())

# double logistics fitting
def double_logi(t, wNDVI,mNDVI,mS,S,mA,A):
    return wNDVI + (mNDVI - wNDVI)* ( 1/(1+np.exp(-mS*(t-S))) + 1/(1+np.exp(mA*(t-A))) - 1)

mod = lmfit.Model(double_logi)

x_ = x4[x4 > 150]
y_ = y4[x4 > 150]

#fitting parameters
params = lmfit.Parameters()
params.add(name = 'wNDVI',value = 0.1,min = 0.1,max = 0.3)
params.add(name = 'mNDVI',value = 0.6,min = 0.6,max = 0.7)
params.add(name = 'mS',value = 0.1,min = 0,max=1)
params.add(name = 'S',value = 150,min = 150,max = 200)
params.add(name = 'mA',value = 0.1,min=0,max=1)
params.add(name = 'A',value = 250,min = 200,max = 300)
result = mod.fit(y_,t = x_,params = params,method="leastsq")    
  
        
# # # #plot the result
result.plot(title = '2022')
chisqr = result.chisqr
plt.text(np.min(x_),np.max(y_),"chisqr = "+ str(round(chisqr,2)))
