# 导入必要的库和模块
import numpy as np # 用于数组和矩阵运算
import pandas as pd # 用于数据处理和分析
# import torch # 用于构建深度学习模型
# import torch.nn as nn # 用于定义模型层

# 定义一个函数，用于将字符串格式的日期转换成整数格式
def date_to_int(date_str):
    # 将日期字符串按照'-'分割成年、月、日三部分
    char = '/' if '/' in date_str else '-'
    year, month, day = date_str.split(char)
    # 将年、月、日转换成整数，并返回
    return int(year), int(month), int(day)

# 定义一个函数，用于将整数格式的日期转换成多维向量
def date_to_vec(date_int):
    # 将日期整数按照年、月、日三部分分别赋值给变量
    year, month, day = date_int
    # 创建一个空列表，用于存储日期向量
    date_vec = []
    # 将年份转换成二进制表示，并添加到列表中
    date_vec.extend([int(x) for x in bin(year)[2:].zfill(12)])
    # 将月份转换成二进制表示，并添加到列表中
    date_vec.extend([int(x) for x in bin(month)[2:].zfill(4)])
    # 将日期转换成二进制表示，并添加到列表中
    date_vec.extend([int(x) for x in bin(day)[2:].zfill(5)])
    # 将列表转换成numpy数组，并返回
    return np.array(date_vec)

# 定义一个函数，用于将字符串格式的时间转换成整数格式
def time_to_int(time_str):
    # 将时间字符串按照':'分割成时、分、秒三部分
    hour, minute, second = time_str.split(':')
    # 将时、分、秒转换成整数，并返回
    return int(hour), int(minute), int(second)

# 定义一个函数，用于将整数格式的时间转换成多维向量
def time_to_vec(time_int):
    # 将时间整数按照时、分、秒三部分分别赋值给变量
    hour, minute, second = time_int
    # 创建一个空列表，用于存储时间向量
    time_vec = []
    # 将小时转换成二进制表示，并添加到列表中
    time_vec.extend([int(x) for x in bin(hour)[2:].zfill(5)])
    # 将分钟转换成二进制表示，并添加到列表中
    time_vec.extend([int(x) for x in bin(minute)[2:].zfill(6)])
    # 将秒转换成二进制表示，并添加到列表中
    time_vec.extend([int(x) for x in bin(second)[2:].zfill(6)])
    # 将列表转换成numpy数组，并返回
    return np.array(time_vec)

# 定义一个函数，用于将字符串格式的日期和时间转换成多维向量
def datetime_to_vec(datetime_str):
    # 将日期和时间字符串按照' '分割成日期部分和时间部分
    date_str, time_str = datetime_str.split(' ')
    # 调用date_to_int函数，将日期字符串转换成整数格式
    date_int = date_to_int(date_str)
    # 调用time_to_int函数，将时间字符串转换成整数格式
    time_int = time_to_int(time_str)
    # 调用date_to_vec函数，将日期整数转换成多维向量
    date_vec = date_to_vec(date_int)
    # 调用time_to_vec函数，将时间整数转换成多维向量
    time_vec = time_to_vec(time_int)
    # 将日期向量和时间向量拼接起来，得到一个更高维的向量，并返回
    return np.concatenate([date_vec, time_vec])

# 定义一个函数，用于将字符串格式的时间序列数据转换成多维向量的数据集
def timeseries_to_dataset(timeseries):
    # 创建一个空列表，用于存储多维向量的数据集
    dataset = []
    # 遍历时间序列数据中的每个元素
    for datetime_str in timeseries:
        # 调用datetime_to_vec函数，将日期和时间字符串转换成多维向量，并添加到列表中
        dataset.append(datetime_to_vec(datetime_str))
    # 将列表转换成numpy数组，并返回
    return np.array(dataset)

# 读取数据集，并将第一列作为索引
df = pd.read_csv('../../data/agriculture_load_h.csv', index_col=0)
# 获取数据集的索引，即日期和时间字符串，并转换成列表
timeseries = df.index.tolist()
# 调用timeseries_to_dataset函数，将时间序列数据转换成多维向量的数据集，并打印形状和内容
dataset = timeseries_to_dataset(timeseries)
print(dataset.shape)
print(dataset)
