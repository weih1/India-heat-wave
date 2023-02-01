# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:46:16 2022

@author: WH
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv(r"G:\urban microclimate\data\India\india\Haryana\climate\association\19_22_climate_ce_twoplace_nor_nonneg.csv")

X = data[['tem','pre','ws']]
y = data['ce']
features = ['tem','pre','ws']

# X = (X -X.min())/(X.max()-X.min())
# y = (y -y.min())/(y.max()-y.min())

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

regressor.feature_importances_

import seaborn as sns
sns.regplot(y_test,y_pred,marker='+',color='black',label = '2019')
plt.scatter(y_test,y_pred)

importances = regressor.feature_importances_
frames = [pd.DataFrame(features), pd.DataFrame(importances)]
feature_importance = pd.concat(frames,axis=1)
feature_importance.columns = ['features','importance']
feature_importance.index=feature_importance['features']
feature_importance.sort_values('importance',inplace = True,ascending=False)
feature_importance['accu_importance'] = feature_importance['importance'].cumsum()


















