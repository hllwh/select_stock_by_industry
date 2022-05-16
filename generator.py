import random

import numpy as np
from tqdm import tqdm
from data_preprocess import *
from utils import *
from plotting import *
from performance import *
from long_short_profoilo import *
import empyrical
import warnings

class generator():

    def __init__(self, settings):
        """
        因子生成器
        :param settings: 参数配置字典
        """

        self.data = settings['data']
        self.func = ['calculate_index','ave_vol_20','ave_vol_60_20','ave_vol_20_to_60',
                     'ave_vol_20_to_250','ave_turnover_20','ave_turnover_60','up_down_rate',
                     'ud_rate_to_vol','asset_group_ratio','return_30_days','return_90_days',
                     'wgt_return_30days','wgt_return_90days','old_Intarday_factor',
                     'new_Intarday_factor','vol_std_30days']
        self.factors = ['ave_vol_20','ave_vol_60_20','ave_vol_20_to_60','ave_vol_20_to_250',
                        'ave_turnover_20','ave_turnover_60','up_down_rate_20','up_down_rate_60_20',
                        'up_down_rate_250_60','ud_rate_to_vol','asset_group_ratio','return_30_days',
                        'return_90_days','wgt_return_30days','wgt_return_90days','old_intarday_factor',
                        'new_intarday_factor','high_r_std_30days','low_r_std_30days','hpl_r_std_30days',
                        'hml_r_std_30days']
        self.combined_factors = ['rank_factor','ICIR_factor','combined_factor_MaxIR','Halflife_IC_factor']
        self.fac_ICIR = {}
        self.fac_MaxIR_weight = {}


    def ave_vol_20(self):
        self.data['ave_vol_20'] = self.data['volume'].rolling(20).mean()

    def ave_vol_60_20(self):
        self.data['ave_vol_60_20'] = (self.data['volume'].rolling(60).sum()-\
                                     self.data['volume'].rolling(20).sum())/40

    def ave_vol_20_to_60(self):
        self.data['ave_vol_20_to_60'] = self.data['volume'].rolling(20).mean()/\
                                           self.data['volume'].rolling(60).mean()

    def ave_vol_20_to_250(self):
        self.data['ave_vol_20_to_250'] = self.data['volume'].rolling(20).mean()/\
                                           self.data['volume'].rolling(250).mean()

    def ave_turnover_20(self):
        self.data['ave_turnover_20'] = self.data['turnover'].rolling(20).mean()

    def ave_turnover_60(self):
        self.data['ave_turnover_60'] = self.data['turnover'].rolling(60).mean()

    def up_down_rate(self):
        self.data['up_down_rate'] = (self.data['close'] -self.data['open'])/\
                                       self.data['open']
        self.data['up_down_rate_20'] = self.data['up_down_rate'].rolling(20).mean()
        self.data['up_down_rate_60_20'] = (self.data['up_down_rate'].rolling(60).sum()-\
                                     self.data['up_down_rate'].rolling(20).sum())/40
        self.data['up_down_rate_250_60'] = (self.data['up_down_rate'].rolling(250).sum() - \
                                              self.data['up_down_rate'].rolling(60).sum()) / 190

    def ud_rate_to_vol(self):
        self.data['ud_rate_to_vol'] = self.data['up_down_rate'].rolling(20).mean()/\
                                         self.data['volume'].rolling(20).mean()

    def special_rate(self):
        return

    #计算动量相关因子

    def return_30_days(self):
        def prod(data):
            return data.prod()-1
        self.data['return_30_days'] = (self.data['day_return'] + 1).rolling(30).apply(prod)

    def return_90_days(self):
        def prod(data):
            return data.prod()-1
        self.data['return_90_days'] = (self.data['day_return'] + 1).rolling(90).apply(prod)

    def wgt_return_30days(self):
        def cal_turnover_weight(data):
            turnover_weight = data['turnover']/data['turnover'].sum()
            return turnover_weight
        wgt_return_30days = []
        for i in tqdm(self.data.index+30):
            turnover_weight = cal_turnover_weight(self.data.iloc[i-30:i])
            wgt_return_30days.append(np.dot(turnover_weight,self.data['day_return'].iloc[i-30:i]))

        wgt_return_30days = [np.nan] * 30 + wgt_return_30days[:-30]
        self.data['wgt_return_30days'] = wgt_return_30days

    def wgt_return_90days(self):
        def cal_turnover_weight(data):
            turnover_weight = data['turnover']/data['turnover'].sum()
            return turnover_weight
        wgt_return_90days = []
        for i in tqdm(self.data.index+90):
            turnover_weight = cal_turnover_weight(self.data.iloc[i-90:i])
            wgt_return_90days.append(np.dot(turnover_weight,self.data['day_return'].iloc[i-90:i]))
        wgt_return_30days = [np.nan] * 90 + wgt_return_90days[:-90]
        self.data['wgt_return_90days'] = wgt_return_90days

    def old_Intarday_factor(self):
        def old_intarday_return(data):
            return data.prod()-1
        self.data = self.data.sort_values(by=['ID','date'])
        self.data['old_intarday_factor'] = (1 + self.data['day_return']).rolling(20).apply(old_intarday_return)

    def new_Intarday_factor(self):
        def new_intarday_return(data):
            data = data.sort_values(by='turnover')
            intarday_return_1 = (1+data['day_return']).iloc[0:5].mean()
            intarday_return_5 = (1 + data['day_return']).iloc[16:20].mean()
            return intarday_return_1,intarday_return_5
        def new_intarday_factor(data):
            data['new_intarday_factor'] = (data['intarday_return_5'] - data['intarday_return_5'].mean())/\
                                          data['intarday_return_5'].std()-(data['intarday_return_1'] - data['intarday_return_1'].mean())/\
                                          data['intarday_return_1'].std()
            return data
        self.data = self.data.sort_values(by=['ID','date'])
        self.data = self.data.reset_index().drop('index',axis=1)
        intarday_return_1_list = []
        intarday_return_5_list = []
        for i in tqdm(self.data.index+20):
            intarday_return_1,intarday_return_5 = new_intarday_return(self.data.iloc[i-20:i])
            intarday_return_1_list.append(intarday_return_1)
            intarday_return_5_list.append(intarday_return_5)
        intarday_return_1_list = [np.nan] * 20 + intarday_return_1_list[:-20]
        intarday_return_5_list = [np.nan] * 20 + intarday_return_5_list[:-20]
        self.data['intarday_return_1'] = intarday_return_1_list
        self.data['intarday_return_5'] = intarday_return_5_list
        group = self.data.groupby(by='date')
        group = group.apply(new_intarday_factor)
        self.data = group.reset_index().drop('index', axis=1)

        #new_intarday_factor  = self.data.rolling(20).apply(lambda x:new_intarday_return(x['return'],x['turnover']))


    # 计算波动率相关因子
    def vol_std_30days(self):
        self.data = self.data.sort_values(by=['ID', 'date'])
        self.data['high_up_rate'] = self.data['high']/self.data['close'].shift(1)
        self.data['low_down_rate'] = self.data['low']/self.data['close'].shift(1)
        self.data['high_r_std_30days'] = self.data['high_up_rate'].rolling(30).std()
        self.data['low_r_std_30days'] = self.data['low_down_rate'].rolling(30).std()
        self.data['hpl_r_std_30days'] = self.data['high_r_std_30days'] + self.data['low_r_std_30days']
        self.data['hml_r_std_30days'] = self.data['high_r_std_30days'] - self.data['low_r_std_30days']

    def calculate_index(self):
        self.data = day_return_calculate(self.data)
        self.data = half_life_weight_calculate(self.data)
        self.data = value_weight_calculate(self.data, 28)
        self.data = day_return_with_value_weight_calculate(self.data)

    def asset_group_ratio(self):
        def rolling_std(data, windows):
            rolling_std = []
            for i in data.index:
                rolling_std.append(data[max(0, i - windows + 1): i + 1].std())
            return rolling_std

        industry_num = 28
        feature_num = 2
        date_interval = 1000
        self.data = asset_group_ratio_process(self.data, feature_num, industry_num, date_interval)
        self.data['asset_group_ratio'] = (self.data['asset_group_ratio'] - self.data['asset_group_ratio'].rolling(date_interval).mean()) \
                                         / rolling_std(self.data['asset_group_ratio'], date_interval)

    def std_factors(self):
        def stdf(data):
            data.loc[:,self.factors] = (data.loc[:,self.factors] - data.loc[:,self.factors].mean())/\
                                       data.loc[:,self.factors].std()
            return data
        group = self.data.groupby(by='date')
        group = group.apply(stdf)
        self.data = group.reset_index().drop('index',axis = 1)

    def combined_factor_bysort(self):
        def rank_factor(data):
            for factor in self.factors:
                rank = data[factor].rank()
                data['rank_factor'] += rank
            return data
        #self.data.fillna()
        self.data['rank_factor'] = 0
        group = self.data.groupby(by='date')
        group = group.apply(rank_factor)
        self.data = group

    def combined_factor_ICIR(self):
        def ICIR_factor(data):
            total_icir = 0
            data['return'] = (data['close'].shift(-1) - data['close'])/data['close']
            for factor in self.factors:
                ic = data[factor].rolling(60).corr(data['return'].shift(-1).rolling(60))
                icir = ic.mean()/ic.std()
                self.fac_ICIR[factor,data['ID'].iloc[0]] = icir
                data['ICIR_factor'] += data[factor] * icir
                total_icir += icir
            data['ICIR_factor'] /= total_icir
            return data

        self.data['ICIR_factor'] = 0
        group = self.data.groupby(by='ID')
        group = group.apply(ICIR_factor)
        self.data = group

    def combined_factor_MaxIR(self):
        IC_Matrix = pd.DataFrame()
        IC_Mean = []
        for factor in self.factors:
            ic = self.data[factor].rolling(60).corr(self.data['return'].shift(-1).rolling(60))
            IC_Matrix[factor] = ic
            IC_Mean.append(ic.mean())
        IC_Matrix = IC_Matrix.fillna(IC_Matrix.mean())
        IC_cov = np.cov(IC_Matrix,rowvar=False)
        IC_cov_inverse = np.linalg.inv(IC_cov)
        weight = np.dot(IC_cov_inverse, np.array(IC_Mean).reshape(len(self.factors), 1))
        weight = pd.DataFrame(weight).apply(lambda x:x[0],axis =1)
        weight = weight/weight.sum()
        combined_factor_MaxIR = np.dot(np.array(self.data[self.factors]),
                                       np.array(weight).reshape(len(self.factors),1))
        self.data['combined_factor_MaxIR'] = combined_factor_MaxIR

    def combined_factor_Halflife_IC(self):
        def half_life_weight(data):
            #print(1)
            H = 2
            N = 30
            weight_list = []
            for i in range(len(data)):
                weight_list.append(2 ** (i - N))
            weighted_ic = np.dot(np.array(data).reshape(1, 30), np.array(weight_list).reshape(30, 1))
            return weighted_ic[0][0]
        self.data['Halflife_IC_factor'] = 0
        total_half_life_IC = [0]*len(self.data)
        for factor in self.factors:
            ic = self.data[factor].rolling(60).corr(self.data['return'].shift(-1).rolling(60))
            #half_life_IC = ic.ewm(span = 30,adjust=False).mean()
            half_life_IC = ic.rolling(30).apply(half_life_weight)
            total_half_life_IC = [i+j for i,j in zip(total_half_life_IC,half_life_IC)]
            self.data['Halflife_IC_factor'] += self.data[factor] * half_life_IC
            print('**************************************************************************')
        self.data['Halflife_IC_factor'] /= total_half_life_IC

        #风险调整的动量因子
        #1.夏普比因子
    def sharpe_factor(self, backdays:int):
        def prod(data):
            return data.prod()-1
        self.data = self.data.sort_values(by=['ID','date'])
        ave_log_return = (1 + self.data['day_return']).rolling(backdays).apply(prod) ** (1/backdays)
        self.data[f'sharpe_factor_{backdays}'] = ave_log_return/(1 + self.data['day_return']).rolling(backdays).std()

    def sharpe_factor2(self, backdays:int):
        def prod(data):
            return data.prod()-1
        self.data = self.data.sort_values(by=['ID','date'])
        ave_return = (1 + self.data['day_return']).rolling(backdays).sum() / backdays
        self.data[f'sharpe_factor2_{backdays}'] = ave_return/(1 + self.data['day_return']).rolling(backdays).std()

        #2信息比率因子
    def Information_ratio_factor(self,backdays:int):
        def Information_ratio(data):
            data['excess_return'] = data['day_return'] - data['day_return'].mean()
            return data
        def prod(data):
            return data.prod()-1
        group = self.data.groupby(by='date')
        group = group.apply(Information_ratio)
        self.data = group.reset_index().drop('index', axis=1)
        self.data = self.data.sort_values(by=['ID', 'date'])
        ave_log_ex_return = (1 + self.data['excess_return']).rolling(backdays).apply(prod) ** (1 / backdays)
        self.data[f'Information_ratio_{backdays}'] = ave_log_ex_return / (1 + self.data['excess_return']).rolling(backdays).std()

    def abs_adjusted_mom(self,backdays:int):
        def abs_adjust(data):
            return (1+data).prod()/(1+np.abs(data)).prod()
        def prod(data):
            return (1+data).prod() - 1
        def abs_prod(data):
            return (1+np.abs(data)).prod() - 1
        self.data = self.data.sort_values(by=['ID', 'date'])
        #self.data[f'abs_adjusted_mom_{backdays}'] = self.data['day_return'].rolling(backdays).apply(abs_adjust)
        self.data[f'abs_adjusted_mom_{backdays}'] = self.data['day_return'].rolling(backdays).apply(prod)/\
                                                    self.data['day_return'].rolling(backdays).apply(abs_prod)

    def max_return_factor(self,backdays:int):
        def max_return(data):
            return data.max()

        self.data = self.data.sort_values(by=['ID', 'date'])
        self.data[f'max_return_factor_{backdays}'] = self.data['day_return'].rolling(backdays).apply(max_return)


    def ID_factor(self,backdays:int):
        def return_change(data):
            pos = np.sum(data>0)/len(data)
            neg = 1 - pos
            return (pos-neg)*abs(data.sum())/data.sum()

        self.data = self.data.sort_values(by=['ID', 'date'])
        self.data[f'ID_factor_{backdays}'] = self.data['day_return'].rolling(backdays).apply(return_change)

    def relative_updown(self,backdays:int):
        def relative_up_down(data):
            pos = data[data>0].mean()
            neg = -data[data<=0].mean()
            return pos/(pos+neg)

        self.data = self.data.sort_values(by=['ID', 'date'])
        self.data[f'relative_updown_{backdays}'] = self.data['day_return'].rolling(backdays).apply(relative_up_down)

    def multi_mom_vol(self,backdays:int):
        def multi_mom_vol(data):
            multi_mom = np.cumprod(1+data)
            return multi_mom.std()

        self.data = self.data.sort_values(by=['ID', 'date'])
        self.data[f'multi_mom_vol_{backdays}'] = self.data['day_return'].rolling(backdays).apply(multi_mom_vol)

if __name__ == '__main__':
    # settings = {'data':None
    #             }
    # settings['data'],a = data_preprocess()
    #
    # g = generator(settings)
    # for factor in g.func:
    #     func = getattr(g, factor, None)
    #     if func is not None:
    #         func()
    # #combined factor
    # g.std_factors()
    # g.combined_factor_ICIR()
    # g.combined_factor_bysort()
    # # factor test
    # #factor_test(g.data, 'ave_vol_20', 'test')
    # print('done')

    settings = {'data':None
                }
    #settings['data'] = pd.read_pickle('./data/all_factor_data/data_with_21+4_factors.pkl')
    #settings['data'] = pd.read_pickle('./data/all_factor_result.pkl')
    settings['data'] = pd.read_csv('./data/g.data2.csv')
    g = generator(settings)
    # g.multi_mom_vol(150)
    # g.relative_updown(10)
    # g.ID_factor(10)
    # g.max_return_factor(10)
    # g.abs_adjusted_mom(10)
    # g.Information_ratio_factor(2)
    # g.sharpe_factor(10)

    # g.multi_mom_vol(30)
    # g.relative_updown(30)
    # g.ID_factor(30)
    # g.max_return_factor(30)
    # #g.abs_adjusted_mom(30)
    # g.Information_ratio_factor(30)
    # g.sharpe_factor(30)
    # g.sharpe_factor2(30)

    #g.new_Intarday_factor()
    g.factors = ['multi_mom_vol_150','relative_updown_10','ID_factor_10','max_return_factor_10',
               'abs_adjusted_mom_10','Information_ratio_10','sharpe_factor_10']
    g.combined_factor_Halflife_IC()
    g.combined_factor_ICIR()
    g.combined_factor_Halflife_IC()
    g.combined_factor_MaxIR()
    g.combined_factor_bysort()

