from data_preprocess import *
from utils import *
from plotting import *

new_data1,new_data2 = data_preprocess()
new_data = new_data1[new_data1['date'] > pd.to_datetime('2017-01-01')]
industry_num = 28
new_data = day_return_calculate(new_data)
new_data = half_life_weight_calculate(new_data)
new_data = value_weight_calculate(new_data,28)
new_data = day_return_with_value_weight_calculate(new_data)

#训练最优参数，feature_num & date_interval
feature_num = [2,3,4,5]#特征向量分别为2，3，4，5，6
date_interval = [125,250,500]#时间间隔为半年，一年，两年
for i in feature_num:
    for j in date_interval:
        data = new_data.copy()
        data = asset_group_ratio_process(data,i,industry_num,j)
        data = asset_group_ratio_std(data,industry_num,j)
        data.to_pickle(f'./result/asp_data/asp_data_of_{i}_feature&_{j}_backdays.pkl')

#可视化处理
for i in feature_num:
    for j in date_interval:
        data = pd.read_pickle(f'./result/asp_data/asp_data_of_{i}_feature&_{j}_backdays.pkl')
        factor_test(data, 'asset_group_ratio', str(i)+'&'+str(j))
        heat_map(data, 'asset_group_ratio', str(i)+'&'+str(j))
