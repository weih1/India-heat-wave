# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:49:08 2022

@author: WH
"""
import pandas as pd
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# read data with LST and TCP
data = pd.read_csv(r"G:\urban microclimate\data\India\india\delhi\分段\2022_4\2022_LST_green_all_tree.csv")
data.index = data['ID']

columns_name = data.columns.values.tolist()
date_cols = columns_name[1:-1]

# linear
def f1(x,a,b):
    return a*x+b

# piecewise linear
def f2(x,a,b,c):
    return a*x*x+b*x+c
        
# fitting effect
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

R2 = pd.DataFrame(np.nan,index = date_cols, columns=[c for c in range(15)],dtype='float')
p1 = pd.DataFrame(np.nan,index = date_cols, columns=[c for c in range(15)],dtype='float')
p2 = pd.DataFrame(np.nan,index = date_cols, columns=[c for c in range(15)],dtype='float')
p3 = pd.DataFrame(np.nan,index = date_cols, columns=[c for c in range(15)],dtype='float')

# data filter and data simplified in 20/30/40 groups
for d in date_cols:
    if data[d][np.isnan(data[d])==False].shape[0] > 100:
        
        x = []
        y = []
        tmp = data.sort_values('greenspace',inplace = False)
        tmp.index = tmp["ID"]
        tmp = tmp[tmp['greenspace'] !=0]
        number = int(len(tmp)/20) #每组多少个
        for i in range(20):
            group = pd.DataFrame(np.nan,columns=['LST','FTC'],index=np.array(tmp['ID'])[i*number:(i+1)*number],dtype='float')
            for j in group.index:
                if (tmp.loc[j][np.isnan(tmp.loc[j]) == False].shape[0] >= 0): #每个id需要超过50天有数据才用/126
                        #print('it is urban')
                    group['FTC'][j] = tmp['greenspace'][j]
                    group['LST'][j] = tmp[d][j]
            FTC = group['FTC'][(np.isnan(group['LST'])==False)] #第d天每个组要超过5个数据
            LST = group['LST'][(np.isnan(group['LST'])==False)]
            if FTC.shape[0] > 5:
                x.append(np.nanmean(FTC))
                y.append(np.nanmean(LST))
        # print (len(x))
        x = np.array(x)
        y = np.array(y)
        # x_1 = x[(np.isnan(x) == False) &  (np.isnan(y)==False)]
        # y_1 = y[(np.isnan(x) == False) &  (np.isnan(y)==False)]
        y_ = y[np.isnan(x)==False]
        x_ = x[np.isnan(x)==False]       

        #循环找断点
        if x_.shape[0] >= 5:  
            for m in range(x_.shape[0] - 5):
                md = m+5     
                def f3(x,a,b,c): 
                    return np.piecewise(x, [x <= x_[md], x > x_[md]], [lambda x:a*x+b, lambda x:a*x+b+c*(x-x_[md])])
              
                p_est3, err_est3 = curve_fit(f3,x,y)
                y_predict_3=f3(x_, *p_est3)
                indexes_3=getIndexes(y_predict_3, y_)
                R2[m][d] = indexes_3[3]
                p1[m][d] = p_est3[0]
                p2[m][d] = p_est3[1]
                p3[m][d] = p_est3[2]
            
R2.to_csv(r"G:\urban microclimate\data\India\india\delhi\分段\treecover\2022\分段拟合\2022_R2.csv")                    
p1.to_csv(r"G:\urban microclimate\data\India\india\delhi\分段\treecover\2022\分段拟合\2022_a.csv")  
p2.to_csv(r"G:\urban microclimate\data\India\india\delhi\分段\treecover\2022\分段拟合\2022_b.csv")  
p3.to_csv(r"G:\urban microclimate\data\India\india\delhi\分段\treecover\2022\分段拟合\2022_c.csv")                       

para = pd.DataFrame(np.nan,index = date_cols, columns=['a','b','c','R2','point_location','point_num'],dtype='float')
for i in R2.index:
    #找到R2最高的断点
    if (R2.loc[i].isnull().all() == False):
        point = R2.loc[i][R2.loc[i] == np.nanmax(R2.loc[i])].index
        para['a'][i] = p1[int(np.array(point))][i]
        para['b'][i] = p2[int(np.array(point))][i]
        para['c'][i] = p3[int(np.array(point))][i]
        para['R2'][i] = R2[int(np.array(point))][i]
        para['point_location'][i] = int(np.array(point))
        para['point_num'][i] = len(R2.loc[i][~np.isnan(R2.loc[i])])+5

para['a1'] = np.nan
para['b1'] = np.nan
para['R2_1'] = np.nan
#对比线性与分段
for d in date_cols:
    if data[d][np.isnan(data[d])==False].shape[0] > 100:
        
        x = []
        y = []
        tmp = data.sort_values('greenspace',inplace = False)
        tmp.index = tmp["ID"]
        tmp = tmp[tmp['greenspace'] !=0]
        number = int(len(tmp)/20) #每组多少个
        for i in range(20):
            group = pd.DataFrame(np.nan,columns=['LST','FTC'],index=np.array(tmp['ID'])[i*number:(i+1)*number],dtype='float')
            for j in group.index:
                if (tmp.loc[j][np.isnan(tmp.loc[j]) == False].shape[0] >= 0): #每个id需要超过50天有数据才用/126
                        #print('it is urban')
                    group['FTC'][j] = tmp['greenspace'][j]
                    group['LST'][j] = tmp[d][j]
            FTC = group['FTC'][(np.isnan(group['LST'])==False)] #第d天每个组要超过5个数据
            LST = group['LST'][(np.isnan(group['LST'])==False)]
            if FTC.shape[0] > 5:
                x.append(np.nanmean(FTC))
                y.append(np.nanmean(LST))
        # print (len(x))
        x = np.array(x)
        y = np.array(y)
        # x_1 = x[(np.isnan(x) == False) &  (np.isnan(y)==False)]
        # y_1 = y[(np.isnan(x) == False) &  (np.isnan(y)==False)]
        y_ = y[np.isnan(x)==False]
        x_ = x[np.isnan(x)==False]
        
        p_est1, err_est1 = curve_fit(f1,x_,y_)
    
        y_predict_1=f1(x_, *p_est1)
    
        indexes_1=getIndexes(y_predict_1, y_)
        para['a1'][d] = p_est1[0]
        para['b1'][d] = p_est1[1]
        para['R2_1'][d] = indexes_1[3]
   
        fig = plt.figure()
        plt.title(d)
        plt.plot(x_,y_,"kx")
        plt.plot(x_, f1(x_, *p_est1), "b--")
        
        
        #分段
        def f3(x):
            return np.piecewise(x, [x <= x_[int(5+para['point_location'][d])], x > x_[int(5+para['point_location'][d])]],
                             [lambda x:para['a'][d]*x+para['b'][d], lambda x:para['a'][d]*x+para['b'][d]+para['c'][d]*(x-x_[int(5+para['point_location'][d])])])
        plt.plot(x_, f3(x_), "r--")
        
        plt.text(np.max(x)-0.1,np.max(y)-0.1,"f1 R2: " + str(round(indexes_1[3],3)))
        plt.text(np.max(x)-0.1,np.max(y)-0.3,"f2 R2: " + str(round(para['R2'][d],3)))

para.to_csv(r"G:\urban microclimate\data\India\india\delhi\分段\treecover\2022\分段拟合\para_tree.csv")

        
#计算CE, CE与分段拟合的规则制定
import pandas as pd
import numpy as np
data = pd.read_csv(r"G:\urban microclimate\data\India\india\delhi\分段\treecover\2019\分段拟合\para_tree.csv")
data.index = data['date']
data['ce']=np.nan
data['class'] = np.nan

#1:单调递减（线性拟合）;2:单调递增（线性拟合）;3:倒V; 4:V; 5:两段递减; 6:两段递增
for i in data.index:
    if (np.isnan(data['R2'][i]) == False):
        if data['R2'][i] < 0.4:
            continue
        else:
            if data['a'][i]*(data['a'][i]+data['c'][i]) < 0:
                data['ce'][i] = np.min([data['a'][i],(data['a'][i]+data['c'][i])])
                if data['a'][i] < 0:
                    data['class'][i] = 4
                else:
                    data['class'][i] = 3
            else:
                if data['a'][i] < 0:
                    if data['R2_1'][i] > 0.4:
                        data['ce'][i] = data['a1'][i]
                        data['class'][i] = 1
                    else:
                        if data['point_location'][i] + 5 > data['point_num'][i]/2:
                            data['ce'][i] = data['a'][i]
                            data['class'][i] = 5
                        else:
                            data['ce'][i] = data['a'][i]+data['c'][i]
                            data['class'][i] = 5
                else:
                    if data['R2_1'][i] > 0.4:
                        data['ce'][i] = data['a1'][i]
                        data['class'][i] = 2
                    else:
                        if data['point_location'][i] + 5 > data['point_num'][i]/2:
                            data['ce'][i] = data['a'][i]
                            data['class'][i] = 6
                        else:
                            data['ce'][i] = data['a'][i]+data['c'][i]
                            data['class'][i] = 6
                        
data.to_csv(r"G:\urban microclimate\data\India\india\delhi\分段\treecover\20\分段拟合\ce_tree.csv",index=None)                                


