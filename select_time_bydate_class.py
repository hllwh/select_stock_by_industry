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

class cross_section_selection_bydate():

    def __init__(self,settings):
        self.raw_data = settings['raw_data']
        self.new_data = settings['new_data']
        self.data = None
        self.subdata = None
        self.factor_direction ={}
        self.map = {}
        self.rank_num = 5
        self.long_list_with_std_factor = None
        self.short_list_with_std_factor = None
        self.short_list = None
        self.long_list = None

    def data_preprocess(self):
        self.new_data = self.new_data[['證券代碼', '簡稱', '年月日', '開盤價(元)', '收盤價(元)', '成交量(千股)', '週轉率％', '市值(百萬元)', '股價淨值比-TEJ']]
        self.new_data.columns = ['ID', 'industry', 'date', 'open', 'close', 'volume', 'turnover', 'value', 'PB']
        self.new_data['ID'] = self.new_data['ID'].apply(lambda x: x[0:5])
        self.new_data['date'] = pd.to_datetime(self.new_data['date'].astype(str)).apply(lambda x:x.date())
        self.new_data['close'] = self.new_data['close'].astype(float)
        self.new_data['volume'] = self.new_data['volume'].astype(float)
        self.new_data['value'] = self.new_data['value'].astype(float)
        self.new_data = self.new_data.sort_values(by='date')
        self.new_data = self.new_data[self.new_data['ID'].apply(lambda x: x not in ['M2300', 'M1700'])]
        datelist = self.raw_data['date'].unique()
        self.new_data = self.new_data[self.new_data['date'].apply(lambda x: x not in datelist)]
        self.data = self.raw_data.append(self.new_data).reset_index().drop('index',axis=1)
        self.data['date'] = pd.to_datetime(self.data['date'].astype(str)).apply(lambda x: x.date())
        self.data.drop_duplicates(subset=['ID', 'date'], keep='first', inplace=True)


    def calculate_index(self):
        self.data = day_return_calculate(self.data)
        self.data = half_life_weight_calculate(self.data)
        self.data = value_weight_calculate(self.data, 28)
        self.data = day_return_with_value_weight_calculate(self.data)

    def calculate_agr(self):
        def rolling_std(data, windows):
            rolling_std = []
            for i in data.index:
                rolling_std.append(data[max(0, i - windows + 1): i + 1].std())
            return rolling_std

        industry_num = 28
        feature_num = 2
        date_interval = 15
        self.data = asset_group_ratio_process(self.data, feature_num, industry_num, date_interval)
        self.data['asset_group_ratio'] = (self.data['asset_group_ratio'] - self.data['asset_group_ratio'].rolling(date_interval).mean()) \
                                         / rolling_std(self.data['asset_group_ratio'], date_interval)

    def get_factor_map(self):
        for ID in self.data['ID'].unique():
            sub_data = self.data[self.data['ID'] == ID]
            coef = sub_data['day_return'].corr(sub_data['asset_group_ratio'])
            coef = coef / abs(coef)
            self.factor_direction[ID] = coef

    def get_indus_map(self):
        map_list = self.raw_data[['ID', 'industry']][-28:]
        for index in map_list.index:
            self.map[map_list.loc[index, 'ID']] = map_list.loc[index, 'industry']

    def long_short_list_with_std_factor(self,date):
        sub_factor_data = self.data[self.data['date'] == date]
        for index in sub_factor_data.index:
            sub_factor_data.loc[index, 'asset_group_ratio'] *= self.factor_direction[sub_factor_data.loc[index, 'ID']]
        # sub_factor_data.apply(change_direction,args=(self.factor_direction,),axis=1)
        sub_factor_data = sub_factor_data.sort_values(by='asset_group_ratio', ascending=False)
        long_list = list(sub_factor_data['ID'][0:self.rank_num].values)
        short_list = list(sub_factor_data['ID'][-self.rank_num:].values)
        long_ind_list = [self.map[i] for i in long_list]
        short_ind_list = [self.map[i] for i in short_list]
        self.long_list_with_std_factor = long_ind_list
        self.short_list_with_std_factor = short_ind_list
        # print('long_ind_list:', long_ind_list)
        # print('long_list:', long_list)
        # print('short_ind_list:', short_ind_list)
        # print('short_list:', short_list)


    def long_short_list(self,date):
        sub_factor_data = self.data[self.data['date'] == date]
        sub_factor_data = sub_factor_data.sort_values(by='asset_group_ratio', ascending=False)
        long_list = list(sub_factor_data['ID'][0:self.rank_num].values)
        short_list = list(sub_factor_data['ID'][-self.rank_num:].values)
        long_ind_list = [self.map[i] for i in long_list]
        short_ind_list = [self.map[i] for i in short_list]
        self.long_list = long_ind_list
        self.short_list = short_ind_list
        # print('long_ind_list:', long_ind_list)
        # print('long_list:', long_list)
        # print('short_ind_list:', short_ind_list)
        # print('short_list:', short_list)


if __name__ == '__main__':
    # settings = {'raw_data':None,
    #             'new_data':None
    #             }
    # settings['raw_data'],a = data_preprocess()
    # settings['new_data'] = pd.read_csv('./data/industry.csv')
    # date = pd.to_datetime('20220408')
    # c = cross_section_selection_bydate(settings)
    # c.data_preprocess()
    # c.calculate_index()
    # c.calculate_agr()
    # c.get_factor_map()
    # c.get_indus_map()
    # c.long_short_list_with_std_factor(date)
    # c.long_short_list(date)

    settings = {'raw_data': None,
                'new_data': None
                }
    settings['raw_data'], a = data_preprocess()
    settings['new_data'] = pd.read_csv('./data/industry.csv')
    c = cross_section_selection_bydate(settings)
    c.data = pd.read_csv('./data/industry_data.csv')
    # c.data_preprocess()
    # c.calculate_index()
    # c.calculate_agr()
    c.get_factor_map()
    c.get_indus_map()
    c.data['date'] = pd.to_datetime(c.data['date'])
    date_list = c.data[c.data['date'] > pd.to_datetime('20210101')]['date'].unique()
    long_short_result = pd.DataFrame(
        columns=['date', 'long_list', 'short_list', 'long_list_with_std_factor', 'long_list_with_std_factor'])
    for index, date in enumerate(date_list):
        c.long_short_list_with_std_factor(date)
        c.long_short_list(date)
        #     result = pd.DataFrame('date':date,'long_list':c.long_list,'short_list':c.short_list,
        #                          'long_list_with_std_factor':c.long_list_with_std_factor,
        #                           'short_list_with_std_factor':c.short_list_with_std_factor)
        long_short_result.loc[index] = [date, c.long_list, c.short_list, c.long_list_with_std_factor,
                                        c.short_list_with_std_factor]



