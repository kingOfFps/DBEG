import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

# matplotlib其实是不支持显示中文的 显示中文需要一行代码设置字体
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 30})
plt.style.use('ggplot')
df=pd.read_csv('../时序数据项目/data/附件1-区域15分钟负荷数据.csv', parse_dates=['数据时间'])
df.info()
data=df.copy()
data=data.set_index('数据时间')
plt.plot(data.index,data['总有功功率（kw）'].values)
# plt.show()

train=data.loc[:'2018/1/13 23:45:00',:]
test=data.loc['2018/1/14 0:00:00':,:]
# 单位根检验-ADF检验
print(sm.tsa.stattools.adfuller(train['总有功功率（kw）']))

# 白噪声检验
acorr_ljungbox(train['总有功功率（kw）'], lags = [6, 12],boxpierce=True)
# 计算ACF
acf=plot_acf(train['总有功功率（kw）'])
plt.title("总有功功率的自相关图")
# plt.show()

# PACF
pacf=plot_pacf(train['总有功功率（kw）'])
plt.title("总有功功率的偏自相关图")
# plt.show()

model = sm.tsa.arima.ARIMA(train,order=(7,0,4))
arima_res=model.fit()
arima_res.summary()
# trend_evaluate = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=20,
#                                             max_ma=5)
# print('train AIC', trend_evaluate.aic_min_order)
# print('train BIC', trend_evaluate.bic_min_order)

predict=arima_res.predict("2018/1/14 0:00:00","2018/1/14 23:45:00")
plt.plot(test.index,test['总有功功率（kw）'])
plt.plot(test.index,predict)
plt.legend(['y_true','y_pred'])
plt.show()
print(len(predict))

from sklearn.metrics import r2_score,mean_absolute_error
mean_absolute_error(test['总有功功率（kw）'],predict)
res=test['总有功功率（kw）']-predict
residual=list(res)
plt.plot(residual)
np.mean(residual)

import seaborn as sns
from scipy import stats
plt.figure(figsize=(10,5))
ax=plt.subplot(1,2,1)
sns.distplot(residual,fit=stats.norm)
ax=plt.subplot(1,2,2)
res=stats.probplot(residual,plot=plt)
plt.show()

predict=arima_res.predict("2018/1/14 0:00:00","2018/1/18 23:45:00")

plt.plot(range(len(predict)),predict)
plt.legend(['y_true','y_pred'])
plt.show()
print(len(predict))




