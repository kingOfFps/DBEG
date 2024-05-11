


def sort_references(references, lang="en"):
    if lang == "zh":
        # Simplified sorting for Chinese, assuming it's already in a comparable format
        return sorted(references, key=lambda x: x)
    else:
        # Sort Western and Russian by the first author's surname
        return sorted(references, key=lambda x: x.split()[0])

# Example usage
references_zh = """
吕志星, 张虓, 王沈征, 等. 基于 K-Means 和 CNN 的用户短期电力负荷预测[J]. 计算机系统应用, 2020, 29(3): 161-166.
钱育树,孔钰婷,黄聪.电力负荷预测研究综述[J].四川电力技术,2023,46(04):37-43+58.DOI:10.16527/j.issn.1003-6954.20230407.
朱彤. 能源转型中我国电力能源的结构, 问题与趋势[J]. 经济导刊, 2020 (6): 48-53.
范英, 衣博文. 能源转型的规律, 驱动机制与中国路径[J]. 管理世界, 2021, 37(8): 95-105.
徐斌, 陈宇芳, 沈小波. 清洁能源发展, 二氧化碳减排与区域经济增长[J]. 经济研究, 2019, 54(7): 188-202.
张希良, 黄晓丹, 张达, 等. 碳中和目标下的能源经济转型路径与政策研究[J]. 管理世界, 2022, 38(1): 35-66.
毛远宏, 孙琛琛, 徐鲁豫, 等. 基于深度学习的时间序列预测方法综述[J]. 微电子学与计算机, 2023, 40(4): 8-17.
梁宏涛, 刘硕, 杜军威, 等. 深度学习应用于时序预测研究综述[J]. Journal of Frontiers of Computer Science & Technology, 2023, 17(6).
王文冠, 沈建冰, 贾云得. 视觉注意力检测综述[J]. 软件学报, 2019, 30(2): 416-439.
"""

references_en = """
Abumohsen M, Owda A Y, Owda M. Electrical load forecasting using LSTM, GRU, and RNN algorithms[J]. Energies, 2023, 16(5): 2283.
Almeshaiei E, Soltan H. A methodology for electric power load forecasting[J]. Alexandria Engineering Journal, 2011, 50(2): 137-144.
Lim B, Zohren S. Time-series forecasting with deep learning: a survey[J]. Philosophical Transactions of the Royal Society A, 2021, 379(2194): 20200209.
Venkatraman A, Hebert M, Bagnell J. Improving multi-step prediction of learned time series models[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2015, 29(1).
Mohajerin N, Waslander S L. Multistep prediction of dynamic systems with recurrent neural networks[J]. IEEE transactions on neural networks and learning systems, 2019, 30(11): 3370-3383.
Din G M U, Marnerides A K. Short term power load forecasting using deep neural networks[C]//2017 International conference on computing, networking and communications (ICNC). IEEE, 2017: 594-598.
Du P, Wang J, Yang W, et al. Multi-step ahead forecasting in electrical power system using a hybrid forecasting system[J]. Renewable energy, 2018, 122: 533-550.
Pirbazari A M, Sharma E, Chakravorty A, et al. An ensemble approach for multi-step ahead energy forecasting of household communities[J]. IEEE Access, 2021, 9: 36218-36240.
Alhussein M, Aurangzeb K, Haider S I. Hybrid CNN-LSTM model for short-term individual household load forecasting[J]. Ieee Access, 2020, 8: 180544-180557.
Fan G F, Peng L L, Hong W C, et al. Electric load forecasting by the SVR model with differential empirical mode decomposition and auto regression[J]. Neurocomputing, 2016, 173: 958-970.
Wang J, Li P, Ran R, et al. A short-term photovoltaic power prediction model based on the gradient boost decision tree[J]. Applied Sciences, 2018, 8(5): 689.
Suthaharan S. Machine learning models and algorithms for big data classification[J]. Integr. Ser. Inf. Syst, 2016, 36: 1-12.
Rigatti S J. Random forest[J]. Journal of Insurance Medicine, 2017, 47(1): 31-39.
Lahouar A, Slama J B H. Hour-ahead wind power forecast based on random forests[J]. Renewable energy, 2017, 109: 529-541.
Askari M, Keynia F. Mid‐term electricity load forecasting by a new composite method based on optimal learning MLP algorithm[J]. IET Generation, Transmission & Distribution, 2020, 14(5): 845-852.
Borghi P H, Zakordonets O, Teixeira J P. A COVID-19 time series forecasting model based on MLP ANN[J]. Procedia Computer Science, 2021, 181: 940-947.
Sherstinsky A. Fundamentals of recurrent neural network (RNN) and long short-term memory (LSTM) network[J]. Physica D: Nonlinear Phenomena, 2020, 404: 132306.
Tokgöz A, Ünal G. A RNN based time series approach for forecasting turkish electricity load[C]//2018 26th Signal processing and communications applications conference (SIU). IEEE, 2018: 1-4.
Veeramsetty V, Reddy K R, Santhosh M, et al. Short-term electric power load forecasting using random forest and gated recurrent unit[J]. Electrical Engineering, 2022, 104(1): 307-329.
Alzubaidi L, Zhang J, Humaidi A J, et al. Review of deep learning: concepts, CNN architectures, challenges, applications, future directions[J]. Journal of big Data, 2021, 8: 1-74.
Imani M. Electrical load-temperature CNN for residential load forecasting[J]. Energy, 2021, 227: 120480.
Lindemann B, Müller T, Vietz H, et al. A survey on long short-term memory networks for time series prediction[J]. Procedia Cirp, 2021, 99: 650-655.
Torres J F, Hadjout D, Sebaa A, et al. Deep learning for time series forecasting: a survey[J]. Big Data, 2021, 9(1): 3-21.
Jahan I S, Snasel V, Misak S. Intelligent systems for power load forecasting: A study review[J]. Energies, 2020, 13(22): 6105.
Zhang W Y, Xie J F, Wan G C, et al. Single-step and multi-step time series prediction for urban temperature based on lstm model of tensorflow[C]//2021 Photonics & Electromagnetics Research Symposium (PIERS). IEEE, 2021: 1531-1535.
Sarkar M R, Anavatti S G, Dam T, et al. Enhancing wind power forecast precision via multi-head attention transformer: An investigation on single-step and multi-step forecasting[C]//2023 International Joint Conference on Neural Networks (IJCNN). IEEE, 2023: 1-8.
Feng B, Xu J, Zhang Y, et al. Multi-step traffic speed prediction based on ensemble learning on an urban road network[J]. Applied Sciences, 2021, 11(10): 4423.
Baržić M, Munitić N F, Bronić F, et al. Forecasting sales in retail with xgboost and iterated multi-step ahead method[C]//2022 International Conference on Smart Systems and Technologies (SST). IEEE, 2022: 153-158.
Hamilton J D. Time series analysis[M]. Princeton university press, 2020.
Wu H, Meng K, Fan D, et al. Multistep short-term wind speed forecasting using transformer[J]. Energy, 2022, 261: 125231.
Mohammadi Farsani R, Pazouki E. A transformer self-attention model for time series forecasting[J]. Journal of Electrical and Computer Engineering Innovations (JECEI), 2020, 9(1): 1-10.
Subakan C, Ravanelli M, Cornell S, et al. Attention is all you need in speech separation[C]//ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021: 21-25.
Abbasimehr H, Paki R. Improving time series forecasting using LSTM and attention models[J]. Journal of Ambient Intelligence and Humanized Computing, 2022, 13(1): 673-691.
Yu Y, Si X, Hu C, et al. A review of recurrent neural networks: LSTM cells and network architectures[J]. Neural computation, 2019, 31(7): 1235-1270.
Smagulova K, James A P. A survey on LSTM memristive neural network architectures and applications[J]. The European Physical Journal Special Topics, 2019, 228(10): 2313-2324.
Siami-Namini S, Tavakoli N, Namin A S. The performance of LSTM and BiLSTM in forecasting time series[C]//2019 IEEE International conference on big data (Big Data). IEEE, 2019: 3285-3292.
Lu W, Li J, Wang J, et al. A CNN-BiLSTM-AM method for stock price prediction[J]. Neural Computing and Applications, 2021, 33(10): 4741-4753.
Siami-Namini S, Tavakoli N, Namin A S. A comparative analysis of forecasting financial time series using arima, lstm, and bilstm[J]. arXiv preprint arXiv:1911.09512, 2019.
Yu J, de Antonio A, Villalba-Mora E. Deep learning (CNN, RNN) applications for smart homes: a systematic review[J]. Computers, 2022, 11(2): 26.
Zhang X, You J. A gated dilated causal convolution based encoder-decoder for network traffic forecasting[J]. IEEE Access, 2020, 8: 6087-6097.
Borovykh A, Bohte S, Oosterlee C W. Dilated convolutional neural networks for time series forecasting[J]. Journal of Computational Finance, Forthcoming, 2018.
Hewage P, Behera A, Trovati M, et al. Temporal convolutional neural (TCN) network for an effective weather forecasting using time-series data from the local weather station[J]. Soft Computing, 2020, 24: 16453-16482.
NIU Z, Zhong G, Yu H. A review on the attention mechanism of deep learning[J]. Neurocomputing, 2021, 452: 48-62.
DE Santana Correia A, Colombini E L. Attention, please! A survey of neural attention models in deep learning[J]. Artificial Intelligence Review, 2022, 55(8): 6037-6124.
Dionisio A, Menezes R, Mendes D A. Mutual information: a measure of dependency for nonlinear time series[J]. Physica A: Statistical Mechanics and its Applications, 2004, 344(1-2): 326-329.
Kumar V, Minz S. Feature selection[J]. SmartCR, 2014, 4(3): 211-229.
Wu K, Wu J, Feng L, et al. An attention‐based CNN‐LSTM‐BiLSTM model for short‐term electric load forecasting in integrated energy system[J]. International Transactions on Electrical Energy Systems, 2021, 31(1): e12637.
Philipp G, Song D, Carbonell J G. The exploding gradient problem demystified-definition, prevalence, impact, origin, tradeoffs, and solutions[J]. arXiv preprint arXiv:1712.05577, 2017.
Jin Y, Guo H, Wang J, et al. A hybrid system based on LSTM for short-term power load forecasting[J]. Energies, 2020, 13(23): 6241.
Dudek G. Multilayer perceptron for short-term load forecasting: from global to local approach[J]. Neural Computing and Applications, 2020, 32(8): 3695-3707.
Zhao Y, Guo N, Chen W, et al. Multi-step ahead forecasting for electric power load using an ensemble model[J]. Expert Systems with Applications, 2023, 211: 118649.
Chandra R, Goyal S, Gupta R. Evaluation of deep learning models for multi-step ahead time series prediction[J]. Ieee Access, 2021, 9: 83105-83123.
Oord A, Dieleman S, Zen H, et al. Wavenet: A generative model for raw audio[J]. arXiv preprint arXiv:1609.03499, 2016.
Liu, R. (2021). Australia Load Dataset [Data set]. figshare. https://doi.org/10.6084/m9.figshare.14958054.v2
Veeramsetty, Venkataramana; Sushma Vaishnavi, Gudelli; Sai Pavan Kumar, Modem; Sumanth, Nagula; Prasanna, Potharaboina (2022), “Electric power load dataset”, Mendeley Data, V1, doi: 10.17632/tj54nv46hj.1
Wu J, Wang Y G, Tian Y C, et al. Support vector regression with asymmetric loss for optimal electric load forecasting[J]. Energy, 2021, 223: 119969.
Nti I K, Teimeh M, Nyarko-Boateng O, et al. Electricity load forecasting: a systematic review[J]. Journal of Electrical Systems and Information Technology, 2020, 7: 1-19.
Hammad M A, Jereb B, Rosi B, et al. Methods and models for electric load forecasting: a comprehensive review[J]. Logist. Sustain. Transp, 2020, 11(1): 51-76.
Sezer O B, Gudelek M U, Ozbayoglu A M. Financial time series forecasting with deep learning: A systematic literature review: 2005–2019[J]. Applied soft computing, 2020, 90: 106181.
Hewage P, Behera A, Trovati M, et al. Temporal convolutional neural (TCN) network for an effective weather forecasting using time-series data from the local weather station[J]. Soft Computing, 2020, 24: 16453-16482.
Liu T, Wang K, Sha L, et al. Table-to-text generation by structure-aware seq2seq learning[C]//Proceedings of the AAAI conference on artificial intelligence. 2018, 32(1).

"""


sorted_refs = sort_references(references_zh)
print("\n".join(sorted_refs))
