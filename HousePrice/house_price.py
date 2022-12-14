# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:53:31 2022

@author: imuka
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split




train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")

x = train_df[['MSSubClass']].values

y = train_df['SalePrice'].values

x_test = test_df[['MSSubClass']].values

#X_train, y_train = train_test_split(X, y, test_size=0.4, random_state=1)

# ランダムフォレスト回帰
forest = RandomForestRegressor(n_estimators=100,
                               criterion='mse', 
                               max_depth=None, 
                               min_samples_split=2, 
                               min_samples_leaf=1, 
                               min_weight_fraction_leaf=0.0, 
                               max_features='auto', 
                               max_leaf_nodes=None, 
                               min_impurity_decrease=0.0, 
                               bootstrap=True, 
                               oob_score=False, 
                               n_jobs=None, 
                               random_state=None, 
                               verbose=0, 
                               warm_start=False, 
                               ccp_alpha=0.0, 
                               max_samples=None
                              )
# モデル学習
forest.fit(x, y)

# 推論
y_train_pred = forest.predict(x)

y_test_pred  = forest.predict(x_test)

x_test_id = test_df['Id'].values
x_test_id = np.array(x_test_id, dtype=np.int32)
print(type(x_test_id[0]))

submit = np.stack([x_test_id, y_test_pred], 1)
print(submit)

submit_pd = pd.DataFrame(submit, columns =['Id','SalePrice'])
submit_pd = submit_pd.astype({'Id': int})

print(submit_pd)

submit_pd.to_csv("myresult.csv", index=False)

""" グラフ可視化 """
# flatten：1次元の配列を返す、argsort：ソート後のインデックスを返す
sort_idx = x.flatten().argsort()

# 可視化用に加工
X_train_plot  = x[sort_idx]
Y_train_plot  = y[sort_idx]
train_predict = forest.predict(X_train_plot)

# 可視化
plt.scatter(X_train_plot, Y_train_plot, color='lightgray', s=70, label='Traning Data')
plt.plot(X_train_plot, train_predict, color='blue', lw=2, label="Random Forest Regression")    

# グラフの書式設定
plt.xlabel('MSSubClass')
plt.ylabel('SalePrice')
plt.legend(loc='upper right')
plt.show()

