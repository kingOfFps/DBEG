import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
存放算法实验过程中，用于处理数据的各个函数。
"""


def getData(filePath: str, ration=0.3):
    if filePath.endswith('.csv'):
        data = pd.read_csv(filePath)
    elif filePath.endswith('.xlsx') or filePath.endswith('.xls'):
        data = pd.read_excel(filePath)
    else:
        raise Exception('请输入csv文件或者excel文件')
    # 删除时间列Timestamp
    data.drop(data.columns[0], axis=1, inplace=True)
    count = int(data.shape[0] * ration)
    # 数据过大，为了节省时间，只部分数据
    return data.iloc[:count, :]


def getData1(filePath: str):
    """和getData()差不多，不过是针对标签在datatime列之后、序列之前的数据集"""
    if filePath.endswith('.csv'):
        data = pd.read_csv(filePath)
    elif filePath.endswith('.xlsx') or filePath.endswith('.xls'):
        data = pd.read_excel(filePath)
    else:
        raise Exception('请输入csv文件或者excel文件')
    # 删除时间列Timestamp
    data.drop(data.columns[0], axis=1, inplace=True)
    """将处于第一列的预测对象吊到最后一列"""
    cols = data.columns.tolist()  # 获取所有列的列表
    cols.append(cols.pop(0))  # 将第一列从列表中取出并添加到列表末尾
    data = data.reindex(columns=cols)  # 调整列的顺序
    # 数据过大，为了节省时间，只取前5000个数据
    # data = data.iloc[:2000, ]
    print(data.shape)
    return data.iloc[:500, :]


# def prepare_data(X, y, stepin, stepout):
#     """数据预处理函数，将普通数据转化为LSTM能训练学习的监督学习格式的数据"""
#     """step:滑动窗口的大小，也就是打算一次用step个数据来预测1个数据，"""
#     newX, newy = [], []
#     for i in range(X.shape[0] - stepin - stepout):
#         """用x[i]~x[i+step-1]来预测y[i+step]"""
#         newX.append(X[i:i + stepin, ])
#         if y.any() is not None:
#             newy.append(y[i + stepin:i + stepin + stepout, ])
#     return np.array(newX), np.array(newy)


def prepare_data(X, y, stepin, stepout):
    """数据预处理函数，将普通数据转化为LSTM能训练学习的监督学习格式的数据"""
    """step:滑动窗口的大小，也就是打算一次用step个数据来预测1个数据，"""
    data = np.concatenate((X, y), axis=1)
    newX, newy = [], []
    for i in range(X.shape[0] - stepin - stepout):
        """用data[i]~data[i+step-1]来预测y[i+step]"""
        newX.append(data[i:i + stepin, ])
        # newX.append(X[i:i + stepin, ])
        if y.any() is not None:
            newy.append(y[i + stepin:i + stepin + stepout, ])
    return np.array(newX), np.array(newy)


def prepare_y(y, stepin, stepout):
    """和prepare_data相似，prepare_data处理的是多特征的数据集。此函数处理的是只含有标签y的数据集"""
    """stepin:滑动窗口的大小，也就是打算一次用step个数据来预测stepout个数据，"""
    newX, newy = [], []
    for i in range(y.shape[0] - stepin - stepout):
        """用x[i]~x[i+step-1]来预测y[i+step]"""
        newX.append(y[i:i + stepin])
        newy.append(y[i + stepin:i + stepin + stepout])
    return np.array(newX), np.array(newy)


# 把数据集切分成训练集，测试集
def splitTrainTest(values, ration):
    values = np.array(values)
    n_train_time = int(values.shape[0] * ration)
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    return train, test


def spliteAndNormalizeXy(trainXy, testXy):
    """对多维数据集进行切分X，y以及归一化"""
    trainX, trainy = trainXy[:, :-1], trainXy[:, -1]
    testX, testy = testXy[:, :-1], testXy[:, -1]
    """归一化：归一化放在数据预处理之后、预测之前是最科学的。因为这样保证了归一化的操作数据，是经过异常处理的相对正常数据"""
    scalerX = MinMaxScaler()
    scalery = MinMaxScaler()
    trainX = scalerX.fit_transform(trainX)
    testX = scalerX.transform(testX)
    trainy = scalery.fit_transform(trainy.reshape(-1, 1))
    testy = scalery.transform(testy.reshape(-1, 1))
    # joblib.dump(scalery, f'../model/scaler/lstm_scalery.pkl')
    return trainX, trainy, testX, testy, scalerX, scalery


def spliteAndNormalizeY(trainy, testy):
    """对单维度及归一化"""
    """归一化：归一化放在数据预处理之后、预测之前是最科学的。因为这样保证了归一化的操作数据，是经过异常处理的相对正常数据"""
    scaler = MinMaxScaler()
    trainy = scaler.fit_transform(trainy)
    # 对测试集testy归一化需要使用训练集的归一化对应的相关参数。因此使用的是transform(),而不需要使用fit_transform()
    testy = scaler.transform(testy)
    # joblib.dump(scaler, f'../model/scaler/lstm_scaler.pkl')
    return trainy, testy, scaler



def deal_data(data: np.ndarray, params, mode='single'):
    """
    """
    assert mode in ['single', 'multiple']
    if mode == 'single':
        scaler = MinMaxScaler()
        # 这里data为imf，为一维的，需要reshape为二维的
        data_scaler = scaler.fit_transform(data.reshape((-1, 1)))
        X, y = prepare_y(data_scaler, params['stepin'], params['stepout'])
        # 切分数据集
        trainX, testX = splitTrainTest(X, params['train_test_ration'])
        trainy, testy = splitTrainTest(y, params['train_test_ration'])
        return trainX, trainy, testX, testy, scaler
    else:
        scalerX = MinMaxScaler()
        scalery = MinMaxScaler()
        X_scaler = scalerX.fit_transform(data[:, :-1])
        y_scaler = scalery.fit_transform(data[:, -1:])
        X, y = prepare_data(X_scaler, y_scaler, params['stepin'], params['stepout'])
        # 切分数据集
        trainX, testX = splitTrainTest(X, params['train_test_ration'])
        trainy, testy = splitTrainTest(y, params['train_test_ration'])
        return trainX, trainy.reshape(-1,1), testX, testy.reshape(-1,1), scalerX, scalery


def deal_data_without_Nom(data: np.ndarray, params, mode='single'):
    """
    """
    assert mode in ['single', 'multiple']
    if mode == 'single':
        scaler = MinMaxScaler()
        # 这里data为imf，为一维的，需要reshape为二维的
        data_scaler = scaler.fit_transform(data.reshape((-1, 1)))
        X, y = prepare_y(data_scaler, params['stepin'], params['stepout'])
        # 切分数据集
        trainX, testX = splitTrainTest(X, params['train_test_ration'])
        trainy, testy = splitTrainTest(y, params['train_test_ration'])
        return trainX, trainy, testX, testy, scaler
    else:
        X = data[:, :-1]
        y = data[:, -1:]
        X, y = prepare_data(X, y, params['stepin'], params['stepout'])
        # 切分数据集
        trainX, testX = splitTrainTest(X, params['train_test_ration'])
        trainy, testy = splitTrainTest(y, params['train_test_ration'])
        return trainX, trainy.reshape(-1,1), testX, testy.reshape(-1,1)

def create_dataset(dataset, look_back):
    '''
    对数据进行处理
    '''
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y


def time_extraction():
    # 读取原始 CSV 文件
    df = pd.read_csv('../../data/agriculture_load_h.csv')
    # 将 'datetime' 列转换为日期时间格式
    df['datetime'] = pd.to_datetime(df['datetime'])
    # 提取年、月、日、小时和分钟作为新的列，并插入到 'datetime' 列后面
    # df.insert(1, 'minute', df['datetime'].dt.minute)
    df.insert(1, 'hour', df['datetime'].dt.hour)
    df.insert(1, 'day', df['datetime'].dt.day)
    df.insert(1, 'month', df['datetime'].dt.month)
    # df.insert(1, 'year', df['datetime'].dt.year)
    # 保存为新的 CSV 文件
    df.to_csv('../../data/agriculture_load_timeExtraction.csv', index=False)

if __name__ == '__main__':
    time_extraction()