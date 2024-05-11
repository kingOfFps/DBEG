from pypinyin import lazy_pinyin as pinyin

references_zh = """
范英, 衣博文. 2021. 能源转型的规律, 驱动机制与中国路径[J]. 管理世界, 37(8): 95-105.
梁宏涛, 刘硕, 杜军威, 胡强, 于旭. 2023. 深度学习应用于时序预测研究综述[J]. Journal of Frontiers of Computer Science & Technology, 17(6).
吕志星, 张虓, 王沈征, 王一, 程思瑾, 秦承龙. 2020. 基于 K-Means 和 CNN 的用户短期电力负荷预测[J]. 计算机系统应用, 29(3), 161-166.
毛远宏, 孙琛琛, 徐鲁豫, 刘曦, 柴波, 贺鹏超. 2023. 基于深度学习的时间序列预测方法综述[J]. 微电子学与计算机, 40(4), 8-17.
钱育树, 孔钰婷, 黄聪. 2023. 电力负荷预测研究综述[J]. 四川电力技术, 46(04): 37-43+58.
孙建梅, 钱秀婷, 王永晴. 2021. 基于改进灰色模型的中长期电力负荷预测[J]. 电工技术, (19), 28-31.
王文冠, 沈建冰, 贾云得. 2019. 视觉注意力检测综述[J]. 软件学报, 30(2): 416-439.
徐斌, 陈宇芳, 沈小波. 2019. 清洁能源发展, 二氧化碳减排与区域经济增长[J]. 经济研究, 54(7): 188-202.
张希良, 黄晓丹, 张达, 耿涌, 田立新, 范英, 陈文颖. 2022. 碳中和目标下的能源经济转型路径与政策研究[J]. 管理世界, 38(1), 35-66.
赵春江. 2024. 智慧农业现状与未来[J]. 山东农机化, 2024(01): 9-10.
朱彤. 2020. 能源转型中我国电力能源的结构, 问题与趋势[J]. 经济导刊, 2020 (6): 48-53.
"""

references_en = """
Abbasimehr H, Paki R. 2022. Improving time series forecasting using LSTM and attention models[J]. Journal of Ambient Intelligence and Humanized Computing, 13(1): 673-691.
Abumohsen M, Owda A Y, Owda M. 2023. Electrical load forecasting using LSTM, GRU, and RNN algorithms[J]. Energies, 16(5): 2283.
Askari M, Keynia F. 2020. Mid-term electricity load forecasting by a new composite method based on optimal learning MLP algorithm[J]. IET Generation, Transmission & Distribution, 14(5): 845-852.
Baržić, M, Munitić N F, Bronić F, Jelić L, Lešić V. 2022 . Forecasting sales in retail with xgboost and iterated multi-step ahead method[C]. International Conference on Smart Systems and Technologies (SST), pp. 153-158
Borovykh A, Bohte S, Oosterlee C W. 2018. Dilated convolutional neural networks for time series forecasting[J]. Journal of Computational Finance, Forthcoming.
Chandra R, Goyal S, Gupta R. 2021. Evaluation of deep learning models for multi-step ahead time series prediction[J]. IEEE Access, 9: 83105-83123.
DE Santana Correia A, Colombini E L. 2022. Attention, please! A survey of neural attention models in deep learning[J]. Artificial Intelligence Review, 55(8): 6037-6124.
Din G M U, Marnerides A K. 2017. Short term power load forecasting using deep neural networks[C]. 2017 International conference on computing, networking and communications (ICNC). IEEE, 2017: 594-598.
Wu H, Gattami A, Flierl M. 2020. Conditional mutual information-based contrastive loss for financial time series forecasting[C]. Proceedings of the First ACM International Conference on AI in Finance. 1-7.
Du P, Wang J, Yang W, Niu T. 2018. Multi-step ahead forecasting in electrical power system using a hybrid forecasting system[J]. Renewable Energy, 122: 533-550.
Dudek G. 2020. Multilayer perceptron for short-term load forecasting: from global to local approach[J]. Neural Computing and Applications, 32(8): 3695-3707.
Feng B, Xu J, Zhang Y, Lin Y. 2021. Multi-step traffic speed prediction based on ensemble learning on an urban road network[J]. Applied Sciences, 11(10): 4423.
Hamilton J D. 2020. Time series analysis[M]. Princeton University Press.
Hammad M A, Jereb B, Rosi B, Dragan D. 2020. Methods and models for electric load forecasting: a comprehensive review[J]. Logistics & Sustainable Transport, 11(1): 51-76.
Hewage P, Behera A, Trovati M, Pereira E, Ghahremani M, Palmieri F, Liu Y. 2020. Temporal convolutional neural (TCN) network for an effective weather forecasting using time-series data from the local weather station[J]. Soft Computing, 24: 16453-16482.
Jahan I S, Snasel V, Misak S. 2020. Intelligent systems for power load forecasting: A study review[J]. Energies, 13(22): 6105.
Jin Y, Guo H, Wang J, Song A. 2020. A hybrid system based on LSTM for short-term power load forecasting[J]. Energies, 13(23): 6241.
Kumar V, Minz S. 2014. Feature selection[J]. SmartCR, 4(3): 211-229.
Lim B, Zohren S. 2021. Time-series forecasting with deep learning: a survey[J]. Philosophical Transactions of the Royal Society A, 379(2194): 20200209.
Lindemann B, Müller T, Vietz H, Jazdi N, Weyrich M. 2021. A survey on long short-term memory networks for time series prediction[J]. Procedia CIRP, 99: 650-655.
Liu R. 2021. Australia Load Dataset [Data set]. figshare. https://doi.org/10.6 084/m9.figshare.1495 8054.v2
Liu T, Wang K, Sha L, Chang B, Sui Z. 2018. Table-to-text generation by structure-aware seq2seq learning[C]. Proceedings of the AAAI conference on artificial intelligence. 32(1).
Lu W, Li J, Wang J, Qin L. 2021. A CNN-BiLSTM-AM method for stock price prediction[J]. Neural Computing and Applications, 33(10): 4741-4753.
Mohajerin N, Waslander S L. 2019. Multistep prediction of dynamic systems with recurrent neural networks[J]. IEEE Transactions on Neural Networks and Learning Systems, 30(11): 3370-3383.
Mohammadi Farsani R, Pazouki E. 2020. A transformer self-attention model for time series forecasting[J]. Journal of Electrical and Computer Engineering Innovations (JECEI), 9(1): 1-10.
NIU Z, Zhong G, Yu H. 2021. A review on the attention mechanism of deep learning[J]. Neurocomputing, 452: 48-62.
Nti I K, Teimeh M, Nyarko-Boateng O, Adekoya A F. 2020. Electricity load forecasting: a systematic review[J]. Journal of Electrical Systems and Information Technology, 7: 1-19.
Oord A, Dieleman S, Zen H, Simonyan K, Vinyals O, Graves A, Kavukcuoglu K. 2016. Wavenet: A generative model for raw audio[J]. arXiv preprint arXiv:1609.03499.
Philipp G, Song D, Carbonell J G. 2017. The exploding gradient problem demystified-definition, prevalence, impact, origin, tradeoffs, and solutions[J]. arXiv preprint arXiv:1712.05577.
Pirbazari A M, Sharma E, Chakravorty A, Elmenreich W,Rong C. 2021. An ensemble approach for multi-step ahead energy forecasting of household communities[J]. IEEE Access, 9: 36218-36240.
Sarkar M R, Anavatti S G, Dam T, Pratama M, Al Kindhi, B. 2023. Enhancing wind power forecast precision via multi-head attention transformer: An investigation on single-step and multi-step forecasting[C]. 2023 International Joint Conference on Neural Networks (IJCNN). IEEE, 2023: 1-8.
Sezer O B, Gudelek M U, Ozbayoglu A M. 2020. Financial time series forecasting with deep learning: A systematic literature review: 2005–2019[J]. Applied Soft Computing, 90: 106181.
Sherstinsky A. 2020. Fundamentals of recurrent neural network (RNN) and long short-term memory (LSTM) network[J]. Physica D: Nonlinear Phenomena, 404: 132306.
Siami-Namini S, Tavakoli N, Namin A S. 2019. A comparative analysis of forecasting financial time series using ARIMA, LSTM, and BiLSTM[J]. arXiv preprint arXiv:1911.09512.
Smagulova K, James A P. 2019. A survey on LSTM memristive neural network architectures and applications[J]. The European Physical Journal Special Topics, 228(10): 2313-2324.
Subakan C, Ravanelli M, Cornell S, Bronzi M, Zhong J. 2021. Attention is all you need in speech separation[C]. ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021: 21-25.
Tokgöz A, Ünal G. 2018. A RNN based time series approach for forecasting turkish electricity load[C]. Signal processing and communications applications conference (SIU). IEEE, 2018: 1-4.
Torres J F, Hadjout D, Sebaa A, Martínez-Álvarez F, Troncoso A. 2021. Deep learning for time series forecasting: a survey[J]. Big Data, 9(1): 3-21.
Veeramsetty V, Reddy K R, Santhosh M, Mohnot A, Singal G. 2022. Short-term electric power load forecasting using random forest and gated recurrent unit[J]. Electrical Engineering, 104(1): 307-329.
Venkatraman A, Hebert M, Bagnell J. 2015. Improving multi-step prediction of learned time series models[C]. Proceedings of the AAAI Conference on Artificial Intelligence. 29(1).
Wu H, Meng K, Fan D, Zhang Z, Liu Q. 2022. Multistep short-term wind speed forecasting using transformer[J]. Energy, 261: 125231.
Wu J, Wang Y G, Tian Y C, Burrage K, Cao T. 2021. Support vector regression with asymmetric loss for optimal electric load forecasting[J]. Energy, 223: 119969.
Wu K, Wu J, Feng L, Yang B, Liang R, Yang S, Zhao R. 2021. An attention-based CNN-LSTM-BiLSTM model for short-term electric load forecasting in integrated energy system[J]. International Transactions on Electrical Energy Systems, 31(1): e12637.
Yu J, de Antonio A, Villalba-Mora E. 2022. Deep learning (CNN, RNN) applications for smart homes: a systematic review[J]. Computers, 11(2): 26.
Yu Y, Si X, Hu C, Zhang J. 2019. A review of recurrent neural networks: LSTM cells and network architectures[J]. Neural Computation, 31(7): 1235-1270.
Zhang Wy, Xie Jf, Wan Gc Tong Ms. 2021. Single-step and multi-step time series prediction for urban temperature based on lstm model of tensorflow[C]. Photonics & Electromagnetics Research Symposium (PIERS). IEEE, 2021: 1531-1535.
Zhang X, You J. 2020. A gated dilated causal convolution based encoder-decoder for network traffic forecasting[J]. IEEE Access, 8: 6087-6097.
Zhao Y, Guo N, Chen W, Zhang H, Guo B, Shen J, Tian Z. 2023. Multi-step ahead forecasting for electric power load using an ensemble model[J]. Expert Systems with Applications, 211: 118649.
Lin J, Ma J, Zhu J,  Cui Y. 2022. Short-term load forecasting based on LSTM networks considering attention mechanism[J]. International Journal of Electrical Power & Energy Systems, 137: 107818.
Du J, Cheng Y, Zhou Q, Zhang J, Zhang X, Li G. 2020. Power load forecasting using BiLSTM-attention[C]. IOP Conference Series: Earth and Environmental Science. IOP Publishing, 440(3): 032115.
Nepal B, Yamaha M, Yokoe A, Yamaji T. 2020. Electricity load forecasting using clustering and ARIMA model for energy management in buildings[J]. Japan Architectural Review, 3(1): 62-76.
Chodakowska E, Nazarko J, Nazarko Ł. 2021. Arima models in electrical load forecasting and their robustness to noise[J]. Energies, 14(23): 7952.
Maulud D, Abdulazeez A M. 2020. A review on linear regression comprehensive in machine learning[J]. Journal of Applied Science and Technology Trends, 1(2): 140-147.
Patil S. 2021. Linear with polynomial regression: Overview[J]. Int J Appl Res, 7: 273-275.
Kuster C, Rezgui Y, Mourshed M. 2017. Electrical load forecasting models: A critical systematic review[J]. Sustainable cities and society, 35: 257-270.
Cortes C, Vapnik V. Support-vector networks[J]. Machine learning, 1995, 20: 273-297.
Klyuev R V, Morgoev I D, Morgoeva A D, Gavrina O A, Martyushev N V, Efremenkov E A, Mengxu Q. 2022. Methods of forecasting electric energy consumption: A literature review[J]. Energies, 15(23): 8919.
Wu J, Wang Y G, Tian Y C, Burrage K, Cao T. 2021. Support vector regression with asymmetric loss for optimal electric load forecasting[J]. Energy, 223: 119969.
Fan Gf, Zhang Lz, Yu M, Hong Wc, Dong S Q. Applications of random forest in multivariable response surface for short-term load forecasting[J]. International Journal of Electrical Power & Energy Systems, 2022, 139: 108073.
Guo W, Che L, Shahidehpour M, et al. Machine-Learning based methods in short-term load forecasting[J]. The Electricity Journal, 2021, 34(1): 106884.
"""

# 中文参考文献按拼音排序
zh_refs = references_zh.strip().split('\n')
# zh_refs_sorted = sorted(zh_refs, key=lambda x: pinyin.get(x, format="strip"))
zh_refs_sorted = sorted(zh_refs, key=lambda x: ''.join(pinyin(x)))
print("\n".join(zh_refs_sorted))
# 英文参考文献按姓氏排序，考虑单个作者和姓氏相同的情况
en_refs = references_en.strip().split('\n')

def sort_en_refs(ref):
    # 分割引用，获取作者部分
    authors = ref.split('.')[0]
    # 获取第一个作者，考虑多个作者的情况
    first_author = authors.split(',')[0]
    # 返回用于排序的键：第一个作者的全名
    return first_author

en_refs_sorted = sorted(en_refs, key=sort_en_refs)
print("\n".join(en_refs_sorted))
# print(len(zh_refs_sorted)+len(en_refs_sorted))

