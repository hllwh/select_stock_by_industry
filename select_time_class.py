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



class cross_section_selection(object):

    def __init__(self, settings):
        """
        执行因子选股策略，对因子值高和低的资产return进行相应增强和削弱
        :param settings: 参数配置字典
        """

        self.factor_data = settings['factor_data']
        self.industry_data = settings['industry_data']
        self.asset_data = settings['asset_data']
        self.rank_num = settings['rank_num']
        self.change_rate = settings['change_rate']
        self.factor_direction = {}

    def map_to_indus(self):
        self.asset_data['ID'] = self.asset_data['asset'].\
            apply(lambda x: self.industry_data[self.industry_data['symbol'] == x]['ID'].values[0])

    def data_preprocess(self):
        self.asset_data = self.asset_data[self.asset_data['datetime'] < '2022/1/11']
        self.asset_data['datetime'] = pd.to_datetime(self.asset_data['datetime'])
        self.factor_data['date'] = pd.to_datetime(self.factor_data['date'])

    def factor_direct(self,factor):
        for ID in self.factor_data['ID'].unique():
            data = self.factor_data[self.factor_data['ID']==ID]
            coef = data['day_return'].corr(data[factor])
            coef = coef/abs(coef)
            self.factor_direction[ID] = coef

    def cross_section_select(self,factor):
        date_list = self.asset_data['datetime'].unique()
        try:
            for date in date_list:
                sub_factor_data = self.factor_data[self.factor_data['date'] == date].\
                    sort_values(by=factor,ascending=False)
                long_list = list(sub_factor_data['ID'][0:self.rank_num].values)
                short_list = list(sub_factor_data['ID'][-self.rank_num:].values)
                sub_data = self.asset_data[self.asset_data['datetime'] == date]
                for rank in range(self.rank_num):
                    sign1 = sub_data.loc[sub_data['ID'].apply(lambda x: x == long_list[rank]), 'ensemble_expected_return']/\
                            abs(sub_data.loc[sub_data['ID'].apply(lambda x: x == long_list[rank]), 'ensemble_expected_return'])
                    sign2 = sub_data.loc[sub_data['ID'].apply(lambda x: x == short_list[-rank-1]), 'ensemble_expected_return']/\
                            abs(sub_data.loc[sub_data['ID'].apply(lambda x: x == short_list[-rank-1]), 'ensemble_expected_return'])
                    sub_data.loc[sub_data['ID'].apply(lambda x: x == long_list[rank]), 'ensemble_expected_return'] *= 1 + self.change_rate[rank] * sign1
                    sub_data.loc[sub_data['ID'].apply(lambda x: x == short_list[-rank-1]), 'ensemble_expected_return'] *= 1 - self.change_rate[rank] * sign2
                self.asset_data.loc[self.asset_data['datetime'] == date, 'ensemble_expected_return'] = sub_data['ensemble_expected_return']
        except:
            print(date)

    def cross_section_select2(self,factor):
        date_list = self.asset_data['datetime'].unique()
        for date in date_list:
            try:
                sub_factor_data = self.factor_data[self.factor_data['date'] == date]
                for index in sub_factor_data.index:
                    sub_factor_data.loc[index,factor] *= self.factor_direction[sub_factor_data.loc[index,'ID']]
                #sub_factor_data.apply(change_direction,args=(self.factor_direction,),axis=1)
                sub_factor_data = sub_factor_data.sort_values(by=factor,ascending=False)
                long_list = list(sub_factor_data['ID'][0:self.rank_num].values)
                short_list = list(sub_factor_data['ID'][-self.rank_num:].values)
                sub_data = self.asset_data[self.asset_data['datetime'] == date]
                for rank in range(self.rank_num):
                    sign1 = sub_data.loc[sub_data['ID'].apply(lambda x: x == long_list[rank]), 'ensemble_expected_return']/\
                            abs(sub_data.loc[sub_data['ID'].apply(lambda x: x == long_list[rank]), 'ensemble_expected_return'])
                    sign2 = sub_data.loc[sub_data['ID'].apply(lambda x: x == short_list[-rank-1]), 'ensemble_expected_return']/\
                            abs(sub_data.loc[sub_data['ID'].apply(lambda x: x == short_list[-rank-1]), 'ensemble_expected_return'])
                    sub_data.loc[sub_data['ID'].apply(lambda x: x == long_list[rank]), 'ensemble_expected_return'] *= 1 + self.change_rate[rank] * sign1
                    sub_data.loc[sub_data['ID'].apply(lambda x: x == short_list[-rank-1]), 'ensemble_expected_return'] *= 1 - self.change_rate[rank] * sign2
                self.asset_data.loc[self.asset_data['datetime'] == date, 'ensemble_expected_return'] = sub_data['ensemble_expected_return']
            except:
                print(date)
                pass
            continue


if __name__ == '__main__':
    settings = {'factor_data': None,
                'industry_data': None,
                'asset_data': None,
                'rank_num': 5,
                'change_rate': [0.15, 0.14, 0.13, 0.12, 0.11]
                }

    settings['factor_data'] = pd.read_pickle('./data/all_factor_result2.pkl')
    settings['industry_data'] = pd.read_csv('./data/industryID.csv')
    settings['asset_data'] = pd.read_csv('./data/final_signal_with_ID.csv')

    stock_selection = cross_section_selection(settings)

    #stock_selection.map_to_indus()
    stock_selection.data_preprocess()
    combined_factors = ['rank_factor', 'ICIR_factor', 'combined_factor_MaxIR', 'Halflife_IC_factor']
    for factor in combined_factors:
        stock_selection.factor_direct(factor)
        stock_selection.cross_section_select2(factor)
        stock_selection.asset_data.to_csv(f'./data/final_signal_result/adjusted_final_sigal_{factor}_0.15_rank5.csv')
    print('all done')
