import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def init():
    # Defaults
    plt.rcParams['figure.figsize'] = (20.0, 10.0)
    plt.rcParams.update({'font.size': 16})
    plt.style.use('ggplot')

def showData(data):
    # Plot the data
    data.plot()
    plt.ylabel('timely airline passengers (x1000)')
    plt.xlabel('Date')
    plt.show()

def arima(dataPath):
    init()
    data = pd.read_csv(dataPath, engine='python',names=['time','cost'],delimiter=",")
    print(data)
    # data['time']=pd.to_datetime(data['time'], format='%Y-%m-%d')
    data['time']=pd.to_datetime(data['time'], format='%Y%m%d')
    data.set_index(['time'], inplace=False)
    # showData(data)

    # Define the d and q parameters to take any value between 0 and 1
    q = d = range(0, 2)
    p = range(0, 4)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    midIndex = int(len(data)*0.8)
    train_data = data[:midIndex]
    test_data = data[midIndex:]

    train_data = data[:"20220514"]
    test_data = data["20220514":]

    warnings.filterwarnings("ignore") # specify to ignore warning messages

    AIC = []
    SARIMAX_model = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            # try:
            print(train_data)
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
            # except:
            #     continue


    print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))


    # Let's fit this model
    mod = sm.tsa.statespace.SARIMAX(train_data,
                                    order=SARIMAX_model[AIC.index(min(AIC))][0],
                                    seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()


    results.plot_diagnostics(figsize=(20, 14))
    plt.show()


    pred0 = results.get_prediction(start='20220514', dynamic=False)
    pred0_ci = pred0.conf_int()


    pred1 = results.get_prediction(start='20220514', dynamic=True)
    pred1_ci = pred1.conf_int()


    pred2 = results.get_forecast('20220520')
    pred2_ci = pred2.conf_int()
    print(pred2.predicted_mean['20220520':])


    ax = data.plot(figsize=(20, 16))
    pred0.predicted_mean.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
    pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
    pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
    ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
    plt.ylabel('timely airline passengers (x1000)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


    prediction = pred2.predicted_mean['1960-01-01':'1960-12-01'].values
    # flatten nested list
    truth = list(itertools.chain.from_iterable(test_data.values))
    # Mean Absolute Percentage Error
    MAPE = np.mean(np.abs((truth - prediction) / truth)) * 100

    print('The Mean Absolute Percentage Error for the forecast of year 1960 is {:.2f}%'.format(MAPE))



"""











    

"""