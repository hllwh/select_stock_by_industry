import matplotlib.pyplot as plt
import pandas as pd
from plotting import *
from performance import *


class Backtest():

    def __init__(self):
        """
        简易回测框架
        :param settings: 参数配置字典
        """

        #回测数据
        self.data = None
        self.returns = None
        self.net_value = []
        self.returns = [0,]
        #回测指标
        self.factoric = None
        self.annual_return = None
        self.annual_vol = None
        self.max_drawdown = None
        self.calmar_ratio = None
        self.sharpe_ratio = None
        self.win_rate = None

    def backtest(self,
                 factor:str,
                 backday:int,
                 fees:float=0):
        self.returns = None
        self.net_value = []
        self.returns = [0,]
        current_cash = 10000
        self.net_value.append(current_cash)
        self.data = self.data.sort_values(by='date')
        date_list = self.data['date'].unique()
        change_position_date = date_list[0:len(date_list):backday]
        self.factoric = np.corrcoef(self.data['day_return'].shift(-1).fillna(method='pad'),
                                    self.data[factor].fillna(self.data[factor].mean()))[0][1]
        #factor_ic = self.data['day_return'].rolling(30).corr(self.data[factor])
        #self.factoric = factor_ic.mean()
        for i,date in enumerate(change_position_date[:-1]):
            data = self.data[self.data['date'] == date]
            data = data.sort_values(by=factor,ascending=(self.factoric<0))
            long_list = data['ID'][0:5]
            short_list = data['ID'][-5:]
            position_data = self.data[(self.data['date']>=date) & (self.data['date']<=change_position_date[i+1])]
            long_return = 0
            short_return = 0
            for long_id in long_list:
                long_return += (1+position_data[position_data['ID']==long_id]['day_return']).prod()-1
            long_return /= 5
            for short_id in short_list:
                short_return += (1+position_data[position_data['ID']==short_id]['day_return']).prod()-1
            short_return /= 5
            current_cash *= 1+long_return-short_return-fees
            self.net_value.append(current_cash)
            self.returns.append(long_return-short_return-fees)
        self.returns = pd.DataFrame(self.returns)
        #self.returns = pd.DataFrame([i/self.net_value[0] for i in self.net_value])

    def calculate_index(self):
        self.annual_return = annual_return(self.returns,0)
        self.annual_vol = annual_volatility(self.returns)
        self.max_drawdown = max_drawdown(self.returns)
        self.calmar_ratio = calmar_ratio(self.returns,0)
        self.sharpe_ratio = sharpe_ratio(self.returns,0)
        self.win_rate = win_rate(self.returns)
        print('annual_return:', self.annual_return.values[0])
        print('annual_volatility:', self.annual_vol.values[0])
        print('max_drawdown:', self.max_drawdown)
        print('calmar_ratio:', self.calmar_ratio.values[0])
        print('sharpe_ratio:', self.sharpe_ratio.values[0])
        print('win_rate', self.win_rate.values[0])

    def plot_net_value(self):
        plt.plot(pd.DataFrame([i/self.net_value[0] for i in self.net_value]))
        plt.show()



if __name__ == '__main__':
    b = Backtest()
    #b.data = pd.read_pickle('./data/all_factor_data/data_with_21+4_factors.pkl')
    #b.data = pd.read_pickle('./data/all_factor_data/data_with_sharpe_factors.pkl')
    b.data = pd.read_csv('./data/g_data.csv')
    b.backtest('Information_ratio_30', 1, fees=0.001)
    b.backtest('sharpe_factor',1,fees=0.002)
    b.calculate_index()
    b.plot_net_value()
    dict = {}
    performance(b.returns,dict,ID = 30,fees = 0)
    print('hi')



