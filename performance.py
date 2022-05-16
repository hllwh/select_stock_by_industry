import numpy as np
import pandas as pd
import empyrical


def annual_return(returns,fees):
    cum_return = (1 + returns).prod() - fees
    num_years = len(returns)/ 252
    return cum_return ** (1/num_years) - 1

def annual_volatility(returns):
    day_std = returns.std()
    return day_std * (252 ** 0.5)

def max_drawdown(returns):
    cum_return = np.cumprod(returns + 1)
    max_drawdown = 100
    for i in cum_return.index:
        if ((cum_return[i+1:]/cum_return.iloc[i]).min() - 1 < max_drawdown).values[0]:
            max_drawdown = ((cum_return[i+1:]/cum_return.iloc[i]).min() - 1).values[0]
    return max_drawdown

def calmar_ratio(returns,fees):
    annual_return_ = annual_return(returns,fees)
    max_drawdown_ = max_drawdown(returns)
    return annual_return_/max_drawdown_

def sharpe_ratio(returns,fees):
    risk_free = 0.03
    annual_return_ = annual_return(returns,fees)
    annual_volatility_ = annual_volatility(returns)
    return (annual_return_ - risk_free)/annual_volatility_

def win_rate(returns):
    return np.sum(returns>0)/(np.sum(returns>0) + np.sum(returns<0))

def performance(returns,dict,ID,fees):
    # annual_return = empyrical.annual_return(returns,period=periods)
    # annual_volatility = empyrical.annual_volatility(returns,period=periods)
    # max_drawdown = empyrical.max_drawdown(returns)
    # calmar = empyrical.calmar_ratio(returns,period=periods)
    # sharpe_ratio = empyrical.sharpe_ratio(returns,risk_free=0.00012,period=periods)
    # win_rate = np.sum(returns>0)/len(returns)
    annual_return_ = annual_return(returns,fees)
    annual_volatility_ = annual_volatility(returns)
    max_drawdown_ = max_drawdown(returns)
    calmar_ratio_ = abs(calmar_ratio(returns,fees))
    sharpe_ratio_ = sharpe_ratio(returns,fees)
    win_rate_ = win_rate(returns)
    # print('annual_return:',annual_return_)
    # print('annual_volatility:',annual_volatility_)
    # print('max_drawdown:', max_drawdown_)
    # print('calmar_ratio:', calmar_ratio_)
    # print('sharpe_ratio:', sharpe_ratio_)
    # print('win_rate', win_rate_)
    dict[ID] = {'annual_return':'{:.3f}'.format(annual_return_.values[0]),
                'annual_volatility':'{:.3f}'.format(annual_volatility_.values[0]),
                'max_drawdown':'{:.3f}'.format(max_drawdown_),
                'calmar_ratio':'{:.3f}'.format(calmar_ratio_.values[0]),
                'sharpe_ratio':'{:.3f}'.format(sharpe_ratio_.values[0]),
                'win_rate': '{:.3f}'.format(win_rate_.values[0]),
                }



