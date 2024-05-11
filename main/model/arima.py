import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import statsmodels.api as sm
import warnings
from main.model.DBEG import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF

def afterTrain(yTrue, yPredict, stepin, isShow):
    # 展示预测结果和真实值
    if isShow:
        showTrueAndPredict1(yTrue, yPredict)
    # 计算各项评价指标
    result = evaluate(yTrue, yPredict)
    filename = os.path.basename(__file__).split('.')[0]
    # saveResult(result, stepin, f'{filename}.csv')

def confirm_p_q(data):
    warnings.filterwarnings('ignore')
    """为了快速训练，max_ar和max_ma取的是1，正常情况应该都取4"""
    AIC = sm.tsa.arma_order_select_ic(data, max_ar=1, max_ma=1, ic='aic')['aic_min_order']
    # AIC = sm.tsa.arma_order_select_ic(data, max_ar=4, max_ma=4, ic='aic')['aic_min_order']
    print(f'AIC:{AIC}')
    print('AIC：', AIC)
    return AIC


def forecast(data, n_step):
    order = confirm_p_q(data)
    model = sm.tsa.ARIMA(data, order=(order[0], 1, order[1]))
    model = model.fit()
    pred = model.forecast(n_step)
    return pred



