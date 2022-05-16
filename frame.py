from data_preprocess import *
from utils import *
from plotting import *
from performance import *
from long_short_profoilo import *
import empyrical
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings

# data loading and proprecess
new_data1,new_data2 = data_preprocess()
new_data = new_data1
industry_num = 28
new_data = day_return_calculate(new_data)
new_data = half_life_weight_calculate(new_data)
new_data = value_weight_calculate(new_data,28)
new_data = day_return_with_value_weight_calculate(new_data)

# calculate asset_group_ratio factor
# 取特征向量为2，计算因子时间间隔为1000days
feature_num = 2
date_interval = 1000
data = new_data.copy()
data = asset_group_ratio_process(data, feature_num, industry_num, date_interval)
data = asset_group_ratio_std(data, industry_num, date_interval)

#截面因子测试
factor_test(data, 'asset_group_ratio', 'test3')
heat_map(data, 'asset_group_ratio', 'test3')

#行业择时策略
data = data[data['date'] > pd.to_datetime('2011-07-05')]
industry_list = data['ID'].unique()
params = pd.DataFrame()
#选取计算周期天数为1250，标准差倍数为1
day = 1250
std_n = 1
for indus in industry_list:
    print(f'result of {indus}:')
    sub_data = data[data['ID'] == indus]
    sub_data = sub_data.reset_index().drop('index',axis = 1)
    params_dict = test_best_params(sub_data, day, std_n, indus)
    params_list = [params_dict['ID'],params_dict['method'],params_dict['direction']] + list(params_dict[indus].values())
    cols = ['ID','method','direction','annual_return','annual_volatility','max_drawdown','calmar_ratio','sharpe_ratio','win_rate']
    params_df = pd.DataFrame(np.array(params_list).reshape(1,9), index=[1], columns=cols)
    # sub_data = long_short2(sub_data,1,day,std_n)
    # returns = sub_data['long_short2'] * sub_data['day_return']
    # performance(returns,dict,indus)
    # cum_return = np.cumprod(1+returns)
    # plt.plot(range(len(cum_return)),cum_return)
    # plt.show()
    params = pd.concat([params,params_df],ignore_index=True)
    print('********************************************************')
    print(params_df)

print(params)