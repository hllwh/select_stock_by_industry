import pandas as pd
import warnings
from utils import *

warnings.filterwarnings('ignore')
def data_preprocess():
    df = pd.read_feather('./data/industry_data.ftr')
    data = df[['證券代碼', '簡稱', '年月日', '開盤價(元)','收盤價(元)','最高價(元)','最低價(元)', '成交量(千股)', '週轉率％','市值(百萬元)', '股價淨值比-TEJ']]
    data.columns = ['ID', 'industry', 'date', 'open', 'close','high','low', 'volume', 'turnover', 'value', 'PB']
    data['date'] = pd.to_datetime(data['date'])
    data['close'] = data['close'].astype(float)
    data['open'] = data['open'].astype(float)
    data['high'] = data['high'].astype(float)
    data['low'] = data['low'].astype(float)
    data['volume'] = data['volume'].astype(float)
    data['value'] = data['value'].astype(float)
    data['turnover'] = data['turnover'].astype(float)
    data['PB'] = data['PB'].astype(float)
    data = data.sort_values(by='date')
    data_20050103_20070629 = data[data['date'] < pd.to_datetime('2007-07-01')]  # 2005-01-03到2007-06-29数据，行业有19个
    data_20070702_20220111 = data[data['date'] > pd.to_datetime('2007-07-01')]  # 2007-07-02到2022-01-11数据，行业有30个
    data_with_28_industry = data_20070702_20220111[data_20070702_20220111['ID'].apply(lambda x: x not in ['M2300', 'M1700'])]
    data_with_20_industry = data_20070702_20220111[data_20070702_20220111['ID'].apply(lambda x: x not in ['M1721', 'M1722', 'M2324', 'M2325', 'M2326', 'M2327', 'M2328', 'M2329', 'M2330', 'M2331'])]
    # data_1701_2202 = data[data['date'] > pd.to_datetime('2017-01-01')] # 2017-01-03到2022-02-11数据，行业有30个
    return data_with_28_industry,data_with_20_industry

def get_opti_data():
    data = pd.read_pickle('./result/asp_data/new_asp_data_of_2_feature&_1000_backdays.pkl')
    data = relative_pb_calculate(data, 1000)
    return data
