import numpy as np
import pandas as pd
from sklearn import decomposition
from tqdm import tqdm
from long_short_profoilo import *
from performance import *
import time


def day_return_calculate(new_data):
    """
    Calculate daily return of n industry index.

    Parameters
    ----------
    new_data : pd.DataFrame
        A dataframe which contains columns of 'ID', 'industry', 'date','open', 'close',
         'volume', 'value', 'PB' and so on.
        'ID': str variables,the symbol of industry index (total number ID is 20 or 28)
        'industry': str variables,the Chinese name of industry index
        'date': datetime variables
        'close': float variables,the close price of industry index
        'volume': float variables,the volume price of industry index
        'value': float variables,the value of industry index
        'PB': float variables,the net value ratio of industry index

    Returns
    -------
    new_data : pd.DataFrame
        new_data with another column:'day_return'
        'day_return': float variables between 0 and 1,the day return of industry index

    """
    new_data = new_data.sort_values(by=['ID', 'date'])
    #new_data['day_return'] = (new_data['close'].shift(-1) - new_data['open'].shift(-1)) / new_data['close']
    # new_data['yes_return_rate'] = ((new_data['close'] - new_data['open']) / new_data['open']).shift(1)
    data_pivot = new_data.pivot('date', 'ID')
    close = data_pivot['close']
    open = data_pivot['open']
    day_return = (close.shift(-1) - open.shift(-1)) / close
    new_data['day_return'] = day_return.unstack().values
    return new_data

def value_weight_calculate(new_data,industry_num):

    """
        Calculate value weight of n industry index.

        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
             'volume', 'value', 'PB' and so on.
            See full explanation in utils.day_return_calculate
        industry_num : int variable, the number of industry, only two values:20 or 28

        Returns
        -------
        new_data : pd.DataFrame
            new_data with another column:'value_weight'
            'value_weight': float variables,the weight value of industry index

        """
    new_data = new_data.sort_values(by='date')
    new_data = new_data.reset_index()
    new_data['sqrt_value'] = np.sqrt(new_data['value'])
    total_sqrt_value = new_data.groupby('date').sum()['sqrt_value']
    new_data['value_weight'] = np.divide(np.array(new_data['sqrt_value']),
                                         np.array(total_sqrt_value[new_data.index // industry_num]))
    return new_data

def half_life_weight_calculate(new_data):

    """
        Calculate half_life_weight of n industry index.

        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
             'volume', 'value', 'PB' and so on.
            See full explanation in utils.day_return_calculate

        Returns
        -------
        new_data : pd.DataFrame
            new_data with another column:'half_life_weight'
            'half_life_weight': float variables,the half life weight of industry index

        """
    new_data = new_data.sort_values(by='date')
    days = len(new_data.groupby('date'))
    base_weight = np.power(0.5, 1 / 125)
    new_data['half_life_weight'] = np.power(base_weight,days-
                                            0.69 * (new_data['date'] - new_data['date'].iloc[0]).
                                            apply(lambda x: x.days))
    return new_data

def day_return_with_value_weight_calculate(new_data):

    """
        Calculate day_return_with_value_weight of n industry index.

        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
             'volume', 'value', 'PB' and so on.
            See full explanation in utils.day_return_calculate

        Returns
        -------
        new_data : pd.DataFrame
            new_data with another column:'weight_day_return'
            'weight_day_return': float variables,the weight day return of industry index

        """
    new_data['weight_day_return'] = np.multiply(np.array(new_data['day_return']),
                                                np.array(new_data['value_weight']))
    return new_data

def asset_group_ratio_calculate(new_data,date_interval,feature_num,industry_num):

    """
        Calculate asset group ratio of n industry index.

        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
             'volume', 'value', 'PB' and so on.
            See full explanation in utils.day_return_calculate
        date_interval: int variable,the number of back-days in calculating covariance matrix
        feature_num : the numbers of feature vectors in PCA method
        industry_num : int variable, the number of industry, only two values:20 or 28

        Returns
        -------
        new_data : pd.DataFrame
            new_data with another column:'asset_group_ratio'
            'asset_group_ratio': float variables,the asset_group_ratio of industry index

        """
    new_data['weight_day_return'] = np.multiply(np.array(new_data['day_return']),
                                                np.array(new_data['weight_value']))
    return new_data

def industry_return_cov_calculate(new_data,index,date_interval,industry_num):

    """
        Calculate covariance and variance matrix of industry index return in past days.

        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
             'volume', 'value', 'PB','weight_return' and so on.
            See full explanation in utils.day_return_calculate
        index: int variable, the index of data
        date_interval: int variable,the number of back-days in calculating covariance matrix
        industry_num : int variable, the number of industry, only two values:20 or 28

        Returns
        -------
        industry_return_cov : pd.DataFrame,covariance of n industry index return during pastdays
        industry_var: series,variance of n industry index return


        """
    industry_return = new_data[index - date_interval * industry_num: index].pivot('date', 'ID')['weight_day_return']
    industry_return_cov = industry_return.cov()
    return industry_return_cov


def absorb_ratio_calculate_by_pca(industry_return_cov: pd.DataFrame, feature_num):

    """
        Calculate absorb_ratio of industry return cov by PCA methods.

        Parameters
        ----------
        industry_return_cov : pd.DataFrame
            covariance of n industry index return during past-days
        feature_num : int variable, the numbers of feature vectors in PCA method
        industry_var: series,variance of n industry index return

        Returns
        -------
        feature_vectors: pd.dataframe,n principal component feature vectors
        absorb_ratio: series,absorb_ratio of n feature vectors


    """
    pca = decomposition.PCA(n_components = feature_num)
    feature_vectors = pca.fit_transform(industry_return_cov)
    feature_vectors = pd.DataFrame(feature_vectors)
    absorb_ratio = pca.explained_variance_ratio_
    return feature_vectors,absorb_ratio

def industry_exposure_to_feature(industry_return_cov, feature_vectors):

    """

        Calculate matrix of industry cov exposure to feature vectors

        Parameters
        ----------
        industry_return_cov : pd.DataFrame
            covariance of n industry index return during past-days
        feature_vectors: array,n principal component feature vectors

        Returns
        -------
        pd.DataFrame(exposure): pd.dataframe,industry_exposure_to_feature vectors

   """
    exposure = np.zeros(shape=(len(industry_return_cov.iloc[0]), len(feature_vectors.iloc[0])))
    for i in range(len(industry_return_cov.iloc[0])):
        for j in range(len(feature_vectors.iloc[0])):
            exposure[i][j] = np.inner(industry_return_cov.iloc[i], feature_vectors.iloc[0:, j])
    return pd.DataFrame(exposure)

def asset_group_ratio_compute(exposure,absorb_ratio,industry_num):

    """
        Calculate asset_group_ratio of n industries in one day

        Parameters
        ----------
        exposure : pd.DataFrame
            industry_exposure_to_feature vectors
        absorb_ratio: series,absorb_ratio of n feature vectors
        industry_num : int variable, the number of industry, only two values:20 or 28

        Returns
        -------
        asset_group_ratio: series,asset_group_ratio of n industries in one day

    """
    asset_group_ratio = []
    total_absort_ratio = absorb_ratio.sum()
    exposure.loc['Col_sum'] = exposure.apply(lambda x: abs(x).sum())
    for i in range(industry_num):
        i_industry_exposure_in_features = exposure.iloc[i]
        #exposure_ratio = np.divide(np.array(i_industry_exposure_in_features),
        #                           np.array(exposure.loc['Col_sum']))
        exposure_ratio = i_industry_exposure_in_features.apply(lambda x:abs(x))/ exposure.loc['Col_sum']
        asset_group_ratio_i = np.inner(exposure_ratio, absorb_ratio)
        asset_group_ratio.append(asset_group_ratio_i/total_absort_ratio)
    return asset_group_ratio

def asset_group_ratio_process(new_data,feature_num,industry_num,date_interval):
    """
        Calculate asset_group_ratio of n industries in one day from oridinary data

        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
            'volume', 'value', 'PB' 'weight_return' and so on.
            See full explanation in utils.day_return_calculate
        feature_num : the numbers of feature vectors in PCA method
        industry_num : int variable, the number of industry, only two values:20 or 28
        date_interval: int variable,the number of back-days in calculating covariance matrix

        Returns
        -------
        new_data : pd.DataFrame
        new_data with another column:'asset_group_ratio'
        'asset_group_ratio': asset_group_ratio of n industries in one day

    """
    new_data['asset_group_ratio'] = np.nan
    new_data['absorb_ratio'] = np.nan
    new_data = new_data.sort_values(by=['date', 'ID'])
    new_data = new_data.reset_index().drop(['index', 'level_0'], 1)
    #new_data = new_data.drop(['index', 'level_0'], 1).reset_index()
    for i in tqdm(range(date_interval * industry_num, len(new_data['ID']) - industry_num + 1, industry_num), ncols=50):
        #time.sleep(1)
        industry_return_cov = industry_return_cov_calculate(new_data, i, date_interval, industry_num)
        feature_vectors, absorb_ratio = absorb_ratio_calculate_by_pca(industry_return_cov, feature_num)
        exposure = industry_exposure_to_feature(industry_return_cov, feature_vectors)
        asset_group_ratio = asset_group_ratio_compute(exposure, absorb_ratio, industry_num)
        new_data.loc[i:i + industry_num - 1, ['asset_group_ratio']] = asset_group_ratio
        new_data.loc[i:i + industry_num - 1, ['absorb_ratio']] = absorb_ratio.sum()
    return new_data

def asset_group_ratio_std(new_data,industry_num,date_interval):
    def rolling_std(data, windows):
        rolling_std = []
        for i in data.index:
            rolling_std.append(data[max(0, i - windows + 1): i + 1].std())
        return rolling_std
    # new_data['asset_group_ratio'].fillna(new_data['asset_group_ratio'].mean(),inplace=True)
    new_data.loc[0:date_interval*industry_num-1, 'asset_group_ratio'] = \
        list(new_data.loc[date_interval*industry_num:2*date_interval*industry_num-1, 'asset_group_ratio'])
    new_data.loc[0:date_interval * industry_num - 1, 'absorb_ratio'] = \
        list(new_data.loc[date_interval * industry_num:2 * date_interval * industry_num - 1, 'absorb_ratio'])
    # 用第二年的数值赋值给第一年(避免算出来标准差为0)
    new_data['asset_group_ratio'] = \
        (new_data['asset_group_ratio'] - new_data['asset_group_ratio'].rolling(
        date_interval).mean()) / rolling_std(new_data['asset_group_ratio'], date_interval)
    return new_data


def relative_pb_calculate(new_data,date_interval):
    """
        Calculate relative pb of n industry index

        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
            'volume', 'value', 'PB' 'weight_return' and so on.
            See full explanation in utils.day_return_calculate
        date_interval: int variable,the number of back-days in calculating mean pb

        Returns
        -------
        new_data : pd.DataFrame
        new_data with another column:'relative_pb'
        'relative_pb': relative pb of n industry index

    """
    pb = new_data.pivot('date', 'ID')['PB']
    pb = pb / pb.rolling(date_interval, min_periods =date_interval//3).mean()
    pb = pb.apply(lambda x: (x - np.mean(x)) / np.std(x))
    relative_pb = pb.div(pb.mean(axis=1), axis='rows')
    new_data['relative_pb'] = relative_pb.unstack().values
    new_data['relative_pb'].fillna(new_data['relative_pb'].mean(), inplace=True)
    return new_data

def div_group_return(data,group_return):
    """
        Calculate return of different groups divided by asp and rpb

        Parameters
        ----------
        data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
            'volume', 'value', 'PB' 'weight_return' and so on.
            See full explanation in utils.day_return_calculate
        group_return: pd.DataFrame,empty

        Returns
        -------
        data : pd.DataFrame, daily return of four groups

    """
    data = data.sort_values(by = 'relative_pb',ascending = False)
    data = data.reset_index().drop('index',1)
    data.loc[0:9,'PB_group'] = 1
    data.loc[10:28,'PB_group'] = 0
    data = data.sort_values(by = 'asset_group_ratio',ascending = False)
    data = data.reset_index().drop('index',1)
    data.loc[0:9, 'agr_group'] = 1
    data.loc[10:28, 'agr_group'] = 0
    for i in range(len(data['ID'])):
        if data.iloc[i]['agr_group'] == 0 and data.iloc[i]['PB_group'] == 0:
            data.loc[i,'group'] = 1
        elif data.iloc[i]['agr_group'] == 1 and data.iloc[i]['PB_group'] == 0:
            data.loc[i,'group'] = 2
        elif data.iloc[i]['agr_group'] == 0 and data.iloc[i]['PB_group'] == 1:
            data.loc[i,'group'] = 3
        elif data.iloc[i]['agr_group'] == 1 and data.iloc[i]['PB_group'] == 1:
            data.loc[i,'group'] = 4
    #计算分组收益
    day_group_return = data.groupby(by='group').sum()['weight_day_return'].reset_index()
    day_group_return['group'] = day_group_return['group'].astype(int)
    date = data['date'][0]
    for i in range(len(day_group_return['group'])):
        group_return.loc[date,day_group_return.loc[i]['group'].astype(int)] = day_group_return.loc[i]['weight_day_return']
    return data

def compare_best_return(indus,dict1,dict2,dict3,dict4):
    """
        Calculate params in which dict will have higher annual_return

    """
    an_return = dict1[indus]['annual_return']
    dict = dict1
    if dict2[indus]['annual_return'] > an_return:
        an_return = dict2[indus]['annual_return']
        dict = dict2
    if dict3[indus]['annual_return'] > an_return:
        an_return = dict3[indus]['annual_return']
        dict = dict3
    if dict4[indus]['annual_return'] > an_return:
        an_return = dict4[indus]['annual_return']
        dict = dict4
    return dict

def test_best_params(sub_data,day,std_n,indus):
    """
        Calculate best params of certain indus

        Parameters
        ----------
        sub_data : pd.DataFrame
        day: days computed in long_short portfolio
        std_n: std_n computed in long_short portfolio

        Returns
        -------
        result : dict, dict of best params, include long-short portfolio num, factor direction
                and performance indexes
    """
    dict1 = {}
    dict1['ID'] = indus
    dict1['method'] = 'long_short1'
    dict1['direction'] = 1
    sub_data1 = sub_data
    sub_data1,fees = long_short1(sub_data1, 1, day, std_n)
    returns = sub_data1['long_short1'] * sub_data1['day_return']
    performance(returns, dict1, indus, 0)
    dict2 = {}
    dict2['ID'] = indus
    dict2['method'] = 'long_short1'
    dict2['direction'] = -1
    sub_data2 = sub_data
    sub_data2,fees = long_short1(sub_data2, -1, day, std_n)
    returns = sub_data2['long_short1'] * sub_data2['day_return']
    performance(returns, dict2, indus, 0)
    dict3 = {}
    dict3['ID'] = indus
    dict3['method'] = 'long_short2'
    dict3['direction'] = 1
    sub_data3 = sub_data
    sub_data3,fees = long_short2(sub_data3, 1, day, std_n)
    returns = sub_data3['long_short2'] * sub_data3['day_return']
    performance(returns, dict3, indus, 0)
    dict4 = {}
    dict4['ID'] = indus
    dict4['method'] = 'long_short2'
    dict4['direction'] = -1
    sub_data4 = sub_data
    sub_data4,fees = long_short2(sub_data4, -1, day, std_n)
    returns = sub_data4['long_short2'] * sub_data4['day_return']
    performance(returns, dict4, indus, 0)
    result = compare_best_return(indus,dict1,dict2,dict3,dict4)
    return result

def change_direction(data,dict):
    data['asset_group_ratio'] = data['asset_group_ratio'] * dict[data['ID']]
    return data
