from data_preprocess import *
from utils import *
from plotting import *

new_data1,new_data2 = data_preprocess()
new_data = new_data1[new_data1['date'] > pd.to_datetime('2017-01-01')]

# pb_data = pd.read_pickle('./result/rpb_data_of_125_backdays.pkl')
# factor_test(pb_data,'relative_pb',125)
#训练最优参数，过去pb均值的天数
date_interval = [125,250,500,1000]
for day in date_interval:
    data = new_data.copy()
    data = relative_pb_calculate(data,day)
    data.to_pickle(f'./result/rpb_data/rpb_data_of_{day}_backdays.pkl')
    factor_test(data,'relative_pb',day)
    heat_map(data,'relative_pb',day)
    print(1)
