import time

from attention import Attention
from keras.layers import Input, Conv1D
from keras.layers import LSTM
from keras.layers.core import *
from keras.layers.merging import concatenate
from keras.models import *
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Dense, Conv1D, Activation, Add, LayerNormalization

from main.model import *
from main.utils.dataProcessing import *
from main.utils.plot import *
from main.utils.utils import *
from main.model.attention import *
from main.model.tcn import *
from main.model.embedding import *


def tcnAttention(params, input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    rate = 0.1
    epsilon = 6
    enc_in, d_model, embed, dropout = input_shape[-1], 128, 'fixed', 0.05
    x1 = DataEmbedding(enc_in, d_model, embed, dropout)(x[:, :, 3:], x[:, :, :3])
    x1 = tf.keras.layers.Dropout(rate)(x1)
    x = Dense(d_model)(x)
    x1 = LayerNormalization(epsilon=epsilon)(x1+x)
    # x = DataEmbedding2(enc_in, d_model,  dropout)(x)

    x2 = attention_block(x1)
    x2 = tf.keras.layers.Dropout(rate)(x2)
    x1 = Dense(d_model)(x1)
    x2 = LayerNormalization(epsilon=epsilon)(x1+x2)

    x3 = tcn_layer(x=x2, output_shape=1,nb_filters=d_model)
    x3 = tf.keras.layers.Dropout(rate)(x3)
    x3 = LayerNormalization(epsilon=epsilon)(x3+x2)
    # 固定的Flatten与全连接层
    x3 = Flatten()(x3)
    output = Dense(1, activation='sigmoid')(x3)
    model = Model(inputs=inputs, outputs=output)
    return model


def afterTrain(yTrue, yPredict, stepin, isShow):
    # 展示预测结果和真实值
    if isShow:
        showTrueAndPredict1(yTrue, yPredict)
    # 计算各项评价指标
    result = evaluate(yTrue, yPredict)
    filename = os.path.basename(__file__).split('.')[0]
    # saveResult(result, stepin, f'{filename}.csv')


def trainAndTest(deal_data_result, isShow=True):
    trainX, trainy, testX, testy = deal_data_result
    # 这里的trainy，testy可能是单维的，也可能是3维度的，需要reshape成2维的（方便训练和画图）
    trainy = trainy.reshape(trainy.shape[0], -1)
    testy = testy.reshape(testy.shape[0], -1)
    n_features = trainX.shape[2]
    # input_shape = trainX.shape[1:]
    input_shape = (params['stepin'], n_features)
    output_shape = 1  # predict one target feature
    """创建模型"""
    # model = create_tcn_model(input_shape=input_shape, output_shape=output_shape)
    model = tcnAttention(params, input_shape)

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(trainX, trainy, epochs=params['epoch'], batch_size=64, validation_split=0.1, verbose=0)
    showTrainLoss(history, isShow=isShow)
    # 预测
    pred = model.predict(testX, verbose=0)
    # 反归一化
    # pred = scalery.inverse_transform(pred)
    # testy = scalery.inverse_transform(testy.reshape(-1, 1))
    # 展示预测结果和真实值
    afterTrain(testy, pred, params['stepin'], isShow)
    return model


"""
SINGLE_ATTENTION_VECTOR用来控制是否使用单一的注意力向量（attention vector），而不是多个注意力向量。
如果为真，那么在计算完注意力权重后，会对注意力权重在时间步长维度上求平均值，得到一个单一的注意力向量，
然后将这个注意力向量重复input_dim次，得到一个二维的注意力矩阵（attention matrix）。
这样做的目的是为了减少计算复杂度和内存消耗，同时保持注意力机制的效果。如果为假，那么就直接使用多个注意力向量作为注意力矩阵。
"""
params = {'epoch': 30, 'units': 64, 'stepin': 20, 'stepout': 1, 'data_ration': 0.2,
          'train_test_ration': 0.8}


def main():
    init()
    data = getData('../../data/agriculture_load_timeExtraction.csv', params['data_ration'])
    deal_data_result = deal_data_without_Nom(data.values, params, mode='multiple')
    # 训练测试
    model = trainAndTest(deal_data_result, isShow=False)


if __name__ == "__main__":
    startTime = time.time()
    for stepin in range(54, 55):
        params['stepin'] = stepin
        print(f'\n{params}')
        main()
    timeCost = round(time.time() - startTime, 1)
    print(f'用时：{timeCost}s')
