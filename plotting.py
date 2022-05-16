import os

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import alphalens_rev as alphalens

def heat_map(new_data,factor,indicator):
    """
        Draw heat map of factor and save as pdf file

        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
            'volume', 'value', 'PB' 'weight_return' and so on.
            This dataframe.columns should contain "factor"
            See full explanation in utils.day_return_calculate
        factor: str variable,factor's name
        indicator: str or int or float variable,indicate different
            for example: different day intervals

        Returns
        -------
    """
    pdf = PdfPages(f'./result/heatmap_pdf/{factor}_heatmap/{factor}_{indicator}.pdf')
    new_data[factor][new_data[factor]<new_data[factor].quantile(0.1)] = new_data[factor].quantile(0.1)
    new_data[factor][new_data[factor]>new_data[factor].quantile(0.9)] = new_data[factor].quantile(0.9)
    heat_data_agr = new_data.pivot('ID', 'date')[factor]
    ax = sns.heatmap(heat_data_agr, cmap = 'RdBu', xticklabels=heat_data_agr.columns, yticklabels=heat_data_agr.index)
    ax.set_title(f'Heatmap for {factor}')  # 图标题
    ax.set_xlabel('date')  # x轴标题
    ax.set_ylabel('asset')
    pdf.savefig()
    plt.show()
    plt.close()
    pdf.close()
    return

def factor_test(new_data,factor,indicator=None):
    """
        Use alphlens to test factor and visualize the result

        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe which contains columns of 'ID', 'industry', 'date', 'close',
            'volume', 'value', 'PB' 'weight_return' and so on.
            This dataframe.columns should contain "factor" and "close"
            See full explanation in utils.day_return_calculate
        factor: str variable,factor's name
        indicator: str or int or float variable,indicate different
            for example: different day intervals

        Returns
        -------
    """
    data_pivot = new_data.pivot('date', 'ID')
    factor_data = data_pivot[factor]
    price = data_pivot['close']
    path = os.getcwd()
    location = path + f'/result1/new_{factor}_{indicator}_test_result'
    os.makedirs(location)
    factor_data_cum = alphalens.utils.get_clean_factor_and_forward_returns(factor_data.stack(),
                                                                                  price,
                                                                                  periods=(1,5),
                                                                                  quantiles=10)
    # factor_data_noncum = alphalens.utils.get_clean_factor_and_forward_returns(factor_data.stack(),
    #                                                                        price,
    #                                                                        periods=(1, 5),
    #                                                                        quantiles=10,
    #                                                                        cumulative_returns=False)
    # print("The result of cumulative_returns:")
    alphalens.tears.create_full_tear_sheet(factor_data_cum, location = location)
    # print("The result of noncumulative_returns:")
    # alphalens.tears.create_full_tear_sheet(factor_data_noncum, location=location)