from module.trans import *

import time

from attention import Attention
from keras.layers import Input, Conv1D
from keras.layers import LSTM
from keras.layers.core import *
from keras.layers.merging import concatenate
from keras.models import *
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Dense, Conv1D, Activation, Add, MultiHeadAttention, \
    GlobalAveragePooling1D, LayerNormalization

from utils.dataProcessing import *
from utils.plot import *
from utils.utils import *
from am.model.attention import *
from am.model.tcn import *

"""
{'epoch': 30, 'units': 64, 'stepin': 54, 'stepout': 1, 'data_ration': 0.2, 'train_test_ration': 0.8}
RMSE: 2935.95	MAPE: 4.15	PCC : 0.97	R2  : 0.94
用时：24.3s
"""


def afterTrain(yTrue, yPredict, stepin, isShow):
    # 展示预测结果和真实值
    if isShow:
        showTrueAndPredict1(yTrue, yPredict)
    # 计算各项评价指标
    result = evaluate(yTrue, yPredict)
    filename = os.path.basename(__file__).split('.')[0]
    saveResult(result, stepin, f'{filename}.csv')


def tcnModel(params, input_shape):
    # 生成模拟数据
    num_samples = 1000
    sequence_length = 10
    num_channels = 5
    d_model = 64
    d_input = sequence_length
    d_channel = num_channels
    d_output = 1
    d_hidden = 512
    num_heads = 3
    num_layers = 2
    dropout = 0.1
    # pe = True
    pe = False

    model = Transformer(d_model, d_input, d_channel, d_output, d_hidden, num_heads, num_layers, dropout, pe)

    return model


def trainAndTest(deal_data_result, isShow=True):
    trainX, trainy, testX, testy, scalerX, scalery = deal_data_result
    # 这里的trainy，testy可能是单维的，也可能是3维度的，需要reshape成2维的（方便训练和画图）
    trainy = trainy.reshape(trainy.shape[0], -1)
    testy = testy.reshape(testy.shape[0], -1)
    n_features = trainX.shape[2]
    # input_shape = trainX.shape[1:]
    input_shape = (params['stepin'], n_features)

    output_shape = 1  # predict one target feature
    # Create the TCN model
    # model = create_tcn_model(input_shape=input_shape, output_shape=output_shape)
    model = tcnModel(params, input_shape)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(trainX, trainy, epochs=params['epoch'], batch_size=64, validation_split=0.1, verbose=0)
    showTrainLoss(history, isShow=isShow)
    # 预测
    pred = model.predict(testX, verbose=0)
    # 反归一化
    pred = scalery.inverse_transform(pred)
    testy = scalery.inverse_transform(testy.reshape(-1, 1))
    # 展示预测结果和真实值
    afterTrain(testy, pred, params['stepin'], isShow)
    return model


params = {'epoch': 15, 'units': 64, 'stepin': 20, 'stepout': 1, 'data_ration': 0.1,
          'train_test_ration': 0.8}


def main():
    init()
    data = getData('../data/agriculture_load_h.csv', params['data_ration'])
    deal_data_result = deal_data(data.values, params, mode='multiple')
    # 训练测试
    model = trainAndTest(deal_data_result, isShow=False)


if __name__ == "__main__":
    startTime = time.time()
    for stepin in range(1, 60):
        params['stepin'] = stepin
        print(f'\n{params}')
        main()
    timeCost = round(time.time() - startTime, 1)
    print(f'用时：{timeCost}s')
