# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:41:12 2022

@author: WH
"""

#piecewise linear fitting of LST and CE
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


#read data with ce and lst
data = pd.read_csv(r"G:\urban microclimate\data\India\india\delhi\分段\treecover\2022\分段拟合\ce_lst_tree.csv")
data.index = data['doy']

#simplified the big data to 20 groups of data in the order of LST
x = []
y = []
tmp = data.sort_values('max',inplace = False)
tmp.index = tmp["doy"]
tmp = tmp[np.isnan(tmp['ce']) ==False]
number = int(len(tmp)/20) #每组多少个
for i in range(20):
    group = pd.DataFrame(np.nan,columns=['ce','LST'],index=np.array(tmp['doy'])[i*number:(i+1)*number],dtype='float')
    for j in group.index:
        group['ce'][j] = np.abs(tmp['ce'][j])
        # group['ce'][j] = tmp['ce'][j]
        group['LST'][j] = tmp['max'][j]
    x.append(np.nanmean(group['LST']))
    y.append(np.nanmean(group['ce']))
        # print (len(x))
# y = np.abs(y)
x = np.array(x)
y = np.array(y)


# the function of fitting effect
def getIndexes(y_predict, y_data):
    n = y_data.size
    # SSE为和方差
    SSE=((y_data-y_predict)**2).sum()
    # MSE为均方差
    MSE=SSE/n
    # RMSE为均方根,越接近0，拟合效果越好
    RMSE=np.sqrt(MSE)
    
    # 求R方，0<=R<=1，越靠近1,拟合效果越好
    u = y_data.mean()
    SST=((y_data-u)**2).sum()
    SSR=SST-SSE
    R_square=SSR/SST
    return SSE, MSE, RMSE, R_square

#export the piecewise linear fitting
result = pd.DataFrame(np.nan,index = [(c+1) for c in range(19)], columns=['split','R2','a','b','c'],dtype='float')
for m in range(x.shape[0] - 1):
    md = m+1    
    def f(l,a,b,c): 
        return np.piecewise(l, [l <= x[md], l > x[md]], [lambda l:a*l+b, lambda l:a*l+b+c*(l-x[md])])
              
    p_est, err_est = curve_fit(f,x,y)
    y_predict=f(x, *p_est)
    indexes=getIndexes(y_predict, y)
    result['split'][md] = md
    result['R2'][md] = indexes[3]
    result['a'][md] = p_est[0]
    result['b'][md] = p_est[1]
    result['c'][md] = p_est[2]
result.index= result['split']
result.to_csv(r"G:\urban microclimate\data\India\india\delhi\分段\treecover\2022\分段拟合\ce_lst_fitting.csv",index=None)    
