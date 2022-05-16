import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings


def long_short1(data,direction,rolling_days,n_std):
    # 建仓平仓策略1：考虑资产集群度变化率作为因子，大于过去10天均值+n_std倍过去10天标准差时建仓买入，小于过去10天均值时平仓
    #                                   小于过去10天均值-n_std倍过去10天标准差建仓卖出，大于过去10天均值时平仓
    # direction表示因子方向，1为正向，-1为反向
    data['apr_change_rate'] = data['asset_group_ratio'] / data['asset_group_ratio'].shift(1) - 1
    data['ave_apr_change_rate'] = data['apr_change_rate'].rolling(rolling_days).mean()
    data['std_apr_change_rate'] = data['apr_change_rate'].rolling(rolling_days).std()
    data['long_short1'] = 0
    position = 0
    total_fees = 0
    fee_rate = 0.003
    base_price = data['close'].iloc[0]
    for i in data.index:
        if position == 0:
            if data.loc[i,'apr_change_rate'] > data.loc[i,'ave_apr_change_rate'] + n_std * data.loc[i,'std_apr_change_rate']:
                position = -1 * direction
                data.loc[i,'long_short1'] = position
                total_fees = total_fees + fee_rate * data.loc[i,'close']/base_price
            elif data.loc[i,'apr_change_rate'] < data.loc[i,'ave_apr_change_rate'] - n_std * data.loc[i,'std_apr_change_rate']:
                position = 1 * direction
                data.loc[i,'long_short1'] = position
                total_fees = total_fees + fee_rate * data.loc[i, 'close'] / base_price
            continue
        elif position == -1 * direction:
            if data.loc[i,'apr_change_rate'] < data.loc[i,'ave_apr_change_rate']:
                position = 0
                total_fees = total_fees + fee_rate * data.loc[i, 'close'] / base_price
            data.loc[i,'long_short1'] = position
            continue
        elif position == 1 * direction:
            if data.loc[i,'apr_change_rate'] > data.loc[i,'ave_apr_change_rate']:
                position = 0
                total_fees = total_fees + fee_rate * data.loc[i, 'close'] / base_price
            data.loc[i,'long_short1'] = position
            continue
    return data,total_fees


def long_short2(data,direction,rolling_days,n_std):
    # 建仓平仓策略，考虑资产集群度因子，大于过去10天均值+过去n_std倍10天标准差时建仓买入，小于过去10天均值时平仓
    #                                   小于过去n_std倍10天均值-过去10天标准差建仓卖出，大于过去10天均值时平仓
    # direction表示因子方向，1为正向，-1为反向
    data['ave_agr'] = data['asset_group_ratio'].rolling(rolling_days).mean()
    data['std_agr'] = data['asset_group_ratio'].rolling(rolling_days).std()
    data['long_short2'] = 0
    position = 0
    total_fees = 0
    fee_rate = 0.003
    base_price = data['close'].iloc[0]
    for i in data.index:
        if position == 0:
            if data.loc[i,'asset_group_ratio'] > data.loc[i,'ave_agr'] + n_std * data.loc[i,'std_agr']:
                position = 1 * direction
                data.loc[i,'long_short2'] = position
                total_fees = total_fees + fee_rate * data.loc[i, 'close'] / base_price
            elif data.loc[i,'asset_group_ratio'] < data.loc[i,'ave_agr'] - n_std * data.loc[i,'std_agr']:
                position = -1 * direction
                data.loc[i,'long_short2'] = position
                total_fees = total_fees + fee_rate * data.loc[i, 'close'] / base_price
            continue
        elif position == 1 * direction:
            if data.loc[i,'asset_group_ratio'] < data.loc[i,'ave_agr']:
                position = 0
                total_fees = total_fees + fee_rate * data.loc[i, 'close'] / base_price
            data.loc[i,'long_short2'] = position
            continue
        elif position == -1 * direction:
            if data.loc[i,'asset_group_ratio'] > data.loc[i,'ave_agr']:
                position = 0
                total_fees = total_fees + fee_rate * data.loc[i, 'close'] / base_price
            data.loc[i,'long_short2'] = position
            continue
    return data,total_fees


def long_short3(data,direction,rolling_days):
    # 建仓平仓策略，考虑资产集群度因子，大于过去10天95%分位数时建仓买入，小于过去10天90%分位数时平仓
    #                                   小于过去10天5%分位数建仓卖出，大于过去10天10%分位数时平仓
    # direction表示因子方向，1为正向，-1为反向
    long_bar = 0.95
    liquidLong_bar = 0.9
    short_bar = 0.05
    liquidshort_bar = 0.1
    data['long_position'] = data['asset_group_ratio'].rolling(rolling_days).quantile(long_bar)
    data['liquidlong_position'] = data['asset_group_ratio'].rolling(rolling_days).quantile(liquidLong_bar)
    data['short_position'] = data['asset_group_ratio'].rolling(rolling_days).quantile(short_bar)
    data['liquidshort_position'] = data['asset_group_ratio'].rolling(rolling_days).quantile(liquidshort_bar)
    data['long_short3'] = 0
    position = 0
    for i in data.index:
        if position == 0:
            if data.loc[i,'asset_group_ratio'] > data.loc[i,'long_position']:
                position = 1 * direction
                data.loc[i,'long_short3'] = position
            elif data.iloc[i]['asset_group_ratio'] < data.iloc[i]['short_position']:
                position = -1 * direction
                data.loc[i,'long_short3'] = position
            continue
        elif position == 1 * direction:
            if data.iloc[i]['asset_group_ratio'] < data.iloc[i]['liquidlong_position']:
                position = 0
            data.loc[i,'long_short3'] = position
            continue
        elif position == -1 * direction:
            if data.iloc[i]['asset_group_ratio'] > data.iloc[i]['liquidshort_position']:
                position = 0
            data.loc[i,'long_short3'] = position
            continue
    return data