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
from select_time_class import *

# #读取计算过(资产集群都)因子的数据
# factor_data = pd.read_pickle('./result1/asp_data/new_asp_data_of_2_feature&_1000_backdays.pkl')
# factor_data = factor_data[factor_data['date'] > pd.to_datetime('2011-07-05')]
#
# #读取个股行业映射数据
# industry = pd.read_csv('./data/industryID.csv')
#
# #读取资产数据
# raw_data = pd.read_csv('./data/final_signal.csv')
# data = raw_data[raw_data['datetime']<'2022/1/11']
#
# #将asset映射到industry
# data['ID'] = data['asset'].apply(lambda x:industry[industry['symbol'] == x]['ID'].values[0])
#
# #截面选股策略
# date_list = data['datetime'].unique()
# for date in date_list:
#     sub_factor_data = factor_data[factor_data['date']==date].sort_values(by='asset_group_ratio',ascending=False)
#     long_list = list(sub_factor_data['ID'][0:5].values)
#     short_list = list(sub_factor_data['ID'][-5:].values)
#     sub_data = data[data['datetime'] == date]
#     sub_data.loc[sub_data['ID'].apply(lambda x: x in long_list),'expected_return'] *=1.1
#     sub_data.loc[sub_data['ID'].apply(lambda x: x in short_list),'expected_return'] *=0.9
#     data.loc[data['datetime'] == date,'expected_return'] = sub_data['expected_return']
#
# #择时选股策略
# indus_list = data['ID'].unique()
# for indus in indus_list:
#     sub_factor_data = factor_data[factor_data['ID']==indus]
#     #取策略2，因子为正向，rollingdays为1250，n_std为1
#     sub_factor_data,fees = long_short2(sub_factor_data,1,1250,1)
#     long_list = list(sub_factor_data[sub_factor_data['long_short2'].apply(lambda x: x == 1)]['date'].values)
#     short_list = list(sub_factor_data[sub_factor_data['long_short2'].apply(lambda x: x == -1)]['date'].values)
#     sub_data = data[data['ID'] == indus]
#     sub_data.loc[sub_data['datetime'].apply(lambda x: x in long_list),'expected_return'] *=1.1
#     sub_data.loc[sub_data['datetime'].apply(lambda x: x in short_list),'expected_return'] *=0.9
#     data.loc[data['ID'] == indus,'expected_return'] = sub_data['expected_return']
#
# print('done')

settings = {'factor_data':None,
            'industry_data':None,
            'asset_data':None,
            'rank_num':5,
            'change_rate':[0.15,0.14,0.13,0.12,0.11]
}

settings['factor_data'] = pd.read_pickle('./result1/asp_data/new_asp_data_of_2_feature&_1000_backdays.pkl')
settings['industry_data'] = pd.read_csv('./data/industryID.csv')
settings['asset_data'] = pd.read_csv('./data/final_signal1.csv')

stock_selection = cross_section_selection(settings)

stock_selection.map_to_indus()
stock_selection.data_preprocess()
stock_selection.factor_direct()
stock_selection.cross_section_select()

stock_selection.asset_data.to_csv('./data/new_adjusted_final_sigal_0.15_rank10.csv')
print('all done')