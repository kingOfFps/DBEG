from .DBEG import tcnAttention_v5
from .ablationExperiment import *
from .contrastExperiment import *
from .models import *


def get_model(modelname, params, input_shape):
    modelnames = ['svm', 'xgboost', 'mlp',
                  'cnn-bilstm-attention', 'attn_twotower']
    allmodelnames = ['svm', 'xgboost', 'mlp',
                     'cnn-bilstm-attention', 'attn_twotower', 'transformer']
    # assert modelname in modelnames
    if modelname == 'xgboost':
        return xgboost_model()
    elif modelname == 'svr':
        return svr_model()
    elif modelname == 'mlp':
        return mlp_model(params, input_shape)
    elif modelname == 'lstm':
        return lstm_model(params, input_shape)
    elif modelname == 'gru':
        return gru_model(params, input_shape)
    elif modelname == 'cnn-attention':
        return cnn_attention(params, input_shape)
    elif modelname == 'cnn-bilstm-attention':
        return cnn_bilstm_attention(params, input_shape)
    elif modelname == 'attn_twotower':
        return tcnAttention_v5(params, input_shape)

    elif modelname == 'StdAttn':
        return stdAttn(params, input_shape)
    elif modelname == 'FeatureBranch':
        return featureBranch(params, input_shape)
    elif modelname == 'TimeBranch':
        return timeBranch(params, input_shape)
    elif modelname == 'NoGate':
        return noGate(params, input_shape)


