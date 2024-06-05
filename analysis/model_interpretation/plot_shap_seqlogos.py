import pandas as pd
import os
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import matplotlib
import scipy
import itertools
import collections
import time

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model, load_model
# import Average
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, GlobalMaxPooling1D, concatenate, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute
from tensorflow.keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply, Average
import tensorflow.keras.backend as K

import shap
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough #this solves the "shap_ADDV2" problem but another one will appear
shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.

from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence

### Helper functions ###

def seq_to_one_hot(seq,order_dict = {'A':0, 'T':3, 'C':1, 'G':2}):
    x = np.zeros((len(seq), 4))
    for (i, bp) in enumerate(seq):
        x[i, order_dict[bp]] = 1
    return x

def seq_to_one_hot_and_pad(seq,pad_len,order_dict = {'A':0, 'T':3, 'C':1, 'G':2}):
    x = np.zeros((len(seq), 4))
    for (i, bp) in enumerate(seq):
        x[i, order_dict[bp]] = 1
    seq_len = len(seq)
    if seq_len < pad_len:
        lpad = (pad_len-seq_len)//2
        rpad = pad_len - seq_len - lpad
        x = np.pad(x,((lpad,rpad),(0,0)),'constant')
    return x

def one_hot_to_seq(x):
    seq = ''
    for i in range(x.shape[0]):
        seq += 'ACGT'[np.argmax(x[i,:])]
    return seq

### Load data ###
cf_dir = 'd3_crossfolds'
# x_tot = np.load(f'{cf_dir}/x_test.npy')
# y_heldout = np.load(f'{cf_dir}/y_test.npy')

# d3_seq_dir = '.'
# d3_seq_df = pd.read_csv(f'{d3_seq_dir}/d3_seq_df_thresh.csv')
# exluded_model_types = ['dhs64_finetuned','dhs62_finetuned','sabetti_ctrl']
# excluded_design_types = ['fsp_minimal_concatemer','concatemer','sabetti_ctrl']
# d3_retraining_df = d3_seq_df[(~d3_seq_df['model_type'].isin(exluded_model_types))&(~d3_seq_df['design_type'].isin(excluded_design_types))].copy()
# pad_len = 200
# x_tot = np.array([seq_to_one_hot_and_pad(seq,pad_len) for seq in d3_retraining_df['sequence']])

d2_seq_dir = '.'
d2_deseq_df = pd.read_csv(f'{d2_seq_dir}/chris_log2fc_df_clean2.csv')
pad_len = 200
x_tot = np.array([seq_to_one_hot_and_pad(seq,pad_len) for seq in d2_deseq_df['enhancer']])

### Define locations and file prefixes ###

model_dir = 'd2_finetuned/test_fold_0/d3_tot'
model_basename = 'd1_wide_ft_cf_t0_v'
model_suffix = '_d2_ft'

results_dir = 'extracted_shap_values_ft/d2_deseq_df'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

### Functions ###

# replaces GlobalMaxPooling1D with MaxPooling1D(pool_size=145), since SHAP doesn't support GlobalMaxPooling1D
def load_model_for_shap(model_dir,model_basename,model_idx,model_suffix=None):

    model = load_model(f'{model_dir}/{model_basename}{model_idx}{model_suffix}.h5')

    n_filters = 608
    filt_sizes = [11,15,21]
    n_dense = 224
    dropout_rate = 0.181
    n_classes = 2
    n_datasets = 2
    lr = 0.0003
    l2_weight = 1e-3

    sequence_input = Input(shape=(200, 4),name="pat_input")

    convs = [None]*len(filt_sizes)

    for i in range(len(filt_sizes)):
        conv1           = Conv1D(n_filters, filt_sizes[i], padding='same', activation='linear', name = "pat_conv_" + str(i))(sequence_input)
        batchnorm1      = BatchNormalization(axis=-1,name = "pat_batchnorm_" + str(i))(conv1)
        exp1            = Activation('exponential',name = "pat_relu_" + str(i))(batchnorm1)
        convs[i]        = Dropout(dropout_rate,name = "pat_dropout_" + str(i))(MaxPooling1D(pool_size=145,name = "pat_pool_" + str(i))(exp1))
        convs[i]        = Lambda(lambda x: K.squeeze(x, axis=1),name = "pat_squeeze_" + str(i))(convs[i])

    concat1           = concatenate(convs,name="pat_concat_layer")

    l2_reg = tf.keras.regularizers.l2(l2_weight)
    dense           = Dense(n_dense,activation='relu',name="pat_dense",kernel_regularizer=l2_reg)(concat1)
    output          = Dense(n_classes,activation='linear',name="pat_output")(dense)

    new_model = Model(inputs=sequence_input, outputs=output)

    # for each conv, batchnorm, and dense layer in model, copy the weights to the same name layer in new_model
    for layer in model.layers:
        # if layer has weights, copy them

        if "conv" in layer.name or "batchnorm" in layer.name or "dense" in layer.name:
            new_model.get_layer(layer.name).set_weights(layer.get_weights())

    new_model.get_layer('pat_output').set_weights(model.get_layer('pat_output').get_weights())

    # set new model to inference mode
    new_model.compile(loss=None, optimizer=tf.keras.optimizers.Adam(lr))

    return new_model

def load_ensemble_model(model_dir,model_basename,model_inds,model_suffix=None):

    # assert isinstance(model_inds, collections.Iterable), "model_inds must be a list or array"

    models = [None] * len(model_inds)
    for i in range(len(model_inds)):
        models[i] = load_model_for_shap(model_dir,model_basename,model_inds[i],model_suffix=model_suffix)
        models[i]._name = f"model_idx{i}"

    ensemble_input = Input(shape=models[0].input_shape[1:])
    ensemble_outputs = [model(ensemble_input) for model in models]
    ensemble_avg = Average()(ensemble_outputs)
    ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_avg)

    return ensemble_model

def shuffle_several_times(s):
    return np.array([dinuc_shuffle(s[0]) for i in range(100)])

### Start interpretations ###

n_crossfolds = 1
n_models = 9

ensemble_inds = np.arange(1,1+n_models)
time_start = time.time()
print('Interpreting ensemble model...')
model = load_ensemble_model(model_dir,model_basename,ensemble_inds,model_suffix=model_suffix)
# explainer = shap.DeepExplainer((model.input,model.output),shuffle_several_times)
# hepg2_raw_shap_explanations, k562_raw_shap_explanations = explainer.shap_values(x_tot,check_additivity=False)

# # save shap explanations
# np.save(os.path.join(results_dir,f'hepg2_raw_shap_explanations_{model_basename}{model_suffix}_ensemble.npy'),hepg2_raw_shap_explanations)
# np.save(os.path.join(results_dir,f'k562_raw_shap_explanations_{model_basename}{model_suffix}_ensemble.npy'),k562_raw_shap_explanations)
# print(f'Shap values for model ensemble model saved.')
# stop_time = time.time()
# print(f'Time elapsed: {stop_time-time_start:.2f} seconds')

# plot sequence logos

# load shap values
hepg2_raw_shap_explanations = np.load(os.path.join(results_dir,f'hepg2_raw_shap_explanations_{model_basename}{model_suffix}_ensemble.npy'))
k562_raw_shap_explanations = np.load(os.path.join(results_dir,f'k562_raw_shap_explanations_{model_basename}{model_suffix}_ensemble.npy'))

seqlogo_dir = 'shap_seqlogos'
os.makedirs(seqlogo_dir,exist_ok=True)

#project the importance at each position onto the base that's actually present
# dinuc_shuff_explanations = np.sum(raw_shap_values,axis=-1)[:,:,None]*x_tot # hmm why sum? maybe this is an error? blegh
dinuc_shuff_explanations = np.sum(hepg2_raw_shap_explanations,axis=-1)[:,:,None]*x_tot - np.sum(k562_raw_shap_explanations,axis=-1)[:,:,None]*x_tot
hepg2_dinuc_shuff_explanations = np.sum(hepg2_raw_shap_explanations,axis=-1)[:,:,None]*x_tot
k562_dinuc_shuff_explanations = np.sum(k562_raw_shap_explanations,axis=-1)[:,:,None]*x_tot

y_tot = model.predict(x_tot)
d2_final_df = pd.read_csv('d2_final_df_qthresh05_v2.csv')

y_meas = d2_deseq_df['log2FoldChange_H2K'].values

for idx in range(d2_deseq_df.shape[0]):
# for idx in [433]:
    seq_idx = idx
    cell_type = d2_deseq_df.loc[seq_idx]['cell_type']
    cur_seq = one_hot_to_seq(x_tot[seq_idx])

    seq_start = 0
    seq_len = 145
    if seq_len < pad_len:
        seq_start = (pad_len-seq_len)//2
    
    dinuc_shuff_explanation = dinuc_shuff_explanations[seq_idx][seq_start:seq_start+seq_len,None]

    # by default calculated as H2K, so if cell type is K562, we need to flip the sign
    if cell_type == 'K562':
        dinuc_shuff_explanation = -1*dinuc_shuff_explanation
    # dinuc_shuff_explanation = hepg2_dinuc_shuff_explanations[seq_idx]
    # dinuc_shuff_explanations = k562_dinuc_shuff_explanations[seq_idx]

    print(f'{seq_idx} : {y_tot[seq_idx,0]-y_tot[seq_idx,1]:.2f} ({y_tot[seq_idx,0]:.2f},{y_tot[seq_idx,1]:.2f}) : {cur_seq}')

    # print(f'Regularized model predicted log2(H/K): {y_test_hat_scaled_avg[seq_idx,0]-y_test_hat_scaled_avg[seq_idx,1]} ({y_test_hat_scaled_avg[seq_idx,0]:.2f},{y_test_hat_scaled_avg[seq_idx,1]:.2f})')

    colors = {0:'#0f9447ff',1:'#235c99ff',2:'#f5b328ff',3:'#d42638ff'}
    fig = plt.figure(figsize=(20,2))
    ax = fig.add_subplot(111)
    viz_sequence.plot_weights_given_ax(ax,dinuc_shuff_explanation, height_padding_factor=0.2,length_padding=1,subticks_frequency=20,highlight={},colors=colors)
    # drop spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # add a horizonal line at a height of 0.9 * upper ylim from start to stop of each row in seq_df
    seq_df = d2_final_df.loc[d2_final_df['sequence_name']==seq_idx]
    # increase upper ylim to make room for horizontal lines
    ax.set_ylim(0,1.1*ax.get_ylim()[1])
    ann_height_prop = 0.85
    text_height_prop = 0.925
    for i in range(seq_df.shape[0]):
        ax.hlines(ann_height_prop*ax.get_ylim()[1],seq_df.iloc[i]['start'],seq_df.iloc[i]['stop'],linewidth=2,color='k')
        # add small vertical lines at start and stop of each row in seq_df, centered at 0.9*upper ylim
        ax.vlines(seq_df.iloc[i]['start'],ann_height_prop*ax.get_ylim()[1]-0.05*ax.get_ylim()[1],ann_height_prop*ax.get_ylim()[1]+0.05*ax.get_ylim()[1],linewidth=2,color='k')
        ax.vlines(seq_df.iloc[i]['stop'],ann_height_prop*ax.get_ylim()[1]-0.05*ax.get_ylim()[1],ann_height_prop*ax.get_ylim()[1]+0.05*ax.get_ylim()[1],linewidth=2,color='k')
        # add the text of motif_alt_id above line, centered
        ax.text((seq_df.iloc[i]['start']+seq_df.iloc[i]['stop'])/2,text_height_prop*ax.get_ylim()[1],seq_df.iloc[i]['motif_alt_id'],ha='center',va='center',fontsize=12)

    # set the xlabel of the plot to be the sequence index, the measured H2K to 2 decimals, and the predicted H2K to 2 decimals
    ax.set_xlabel(f'Seq. {seq_idx}: meas. log2(HEPG2/K562) = ${y_meas[seq_idx]:.2f}$, pred. log2(HEPG2/K562) = ${y_tot[seq_idx,0]-y_tot[seq_idx,1]:.2f}$',fontsize=14)
    # layel y axis Delta SHAP, using the greek symbol for Delta
    ax.set_ylabel(r'$\Delta$SHAP',fontsize=14)

    # save the figure
    plt.savefig(f'{seqlogo_dir}/d2_seq_{seq_idx}_delta_shap.png',format='png',dpi=400,bbox_inches='tight')
    plt.close()

    # plt.show()

# d2