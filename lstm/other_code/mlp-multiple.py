import time
from keras.layers import Dense,Flatten
from keras.models import Sequential
from keras import backend as K

from utils.dataProcessing import *
from utils.plot import *
from utils.utils import *

"""利用LSTM对数据进行多特征预测。"""



def trainAndTest(deal_data_result, isShow=True):
    trainX, trainy, testX, testy, scalerX,scalery = deal_data_result
    # 这里的trainy，testy可能是单维的，也可能是3维度的，需要reshape成2维的（方便训练和画图）
    trainy = trainy.reshape(trainy.shape[0], -1)
    testy = testy.reshape(testy.shape[0], -1)
    n_features = trainX.shape[2]
    model = Sequential()
    # params['stepIn']就是滑动窗口的长度（不算y）
    model.add(Dense(params['units'], activation='relu', input_shape=(params['stepIn'], n_features)))
    model.add(Flatten())
    model.add(Dense(trainy.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    """画出训练过程的loss变化"""
    history = model.fit(trainX, trainy, validation_data=(trainX, trainy), epochs=params['epoch'], batch_size=64,
                        shuffle=True, verbose=0)
    # 展示训练过程中loss的变化
    showTrainLoss(history, isShow=isShow)
    # 预测
    pred = model.predict(testX, verbose=0)
    pred = pred.reshape((pred.shape[0],-1))

    # 反归一化
    pred = scalery.inverse_transform(pred)
    testy = scalery.inverse_transform(testy)
    # 展示预测结果和真实值
    if isShow:
        showTrueAndPredict1(testy, pred)
    # 计算各项评价指标
    result = evaluate(testy, pred)
    filename = os.path.basename(__file__).split('.')[0]
    saveResult(result, params['stepIn'], f'{filename}.csv')
    return model


def deal_data(data: np.ndarray):
    """
    0. 归一化（归一化放在预测之前是最科学的）
    1. 切分训练集、测试集
    2. 滑动窗口构造数据集
    :param data:从csv中读取的数据集
    """
    scalerX = MinMaxScaler()
    scalery = MinMaxScaler()
    X_scaler = scalerX.fit_transform(data[:,:-1])
    y_scaler = scalery.fit_transform(data[:,-1:])
    X, y = prepare_data(X_scaler,y_scaler, params['stepIn'], params['stepOut'])
    # 切分数据集
    trainX, testX = splitTrainTest(X, params['ration'])
    trainy, testy = splitTrainTest(y, params['ration'])
    return trainX, trainy, testX, testy, scalerX,scalery


def main():
    init()
    data = getData('../data/agriculture_load_h.csv')
    deal_data_result = deal_data(data.values)
    # 训练测试
    model = trainAndTest(deal_data_result, isShow=True)

"""
stepIn:相当于一次用多少个样本来预测，也就是滑动窗口的大小
stepOut: 相当于一次预测多少步y。目前的代码逻辑只能选择stepOut=1
"""
# params = {'epoch': 30, 'units': 50, 'stepIn': 2, 'stepOut': 1, 'ration': 0.8}
params = {'epoch': 20, 'units': 64, 'stepIn': 3, 'stepOut': 1, 'ration': 0.8}

if __name__ == "__main__":
    startTime = time.time()
    for stepin in range(3, 3+1):
        params['stepIn'] = stepin
        print(f'\n{params}')
        main()
    timeCost = round(time.time() - startTime, 1)
    print(f'用时：{timeCost}s')

"""
{'epoch': 15, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}
Test RMSE: 8898.4709
Test MAPE: 12.9422
皮尔森系数0.73
决定系数0.45
用时：2.2s

{'epoch': 15, 'units': 64, 'stepIn': 2, 'stepOut': 1, 'ration': 0.8}
Test RMSE: 8795.4589
Test MAPE: 12.4407
皮尔森系数0.7
决定系数0.46
用时：2.3s

"""
