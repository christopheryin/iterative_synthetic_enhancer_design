import pandas as pd
import os
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, ranksums, linregress
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import itertools
import statannot
from statsmodels.stats.multitest import fdrcorrection

from deeplift.visualization import viz_sequence


matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'


HEPG2_COL = 'log2FoldChange_HepG2_DNA'
K562_COL = 'log2FoldChange_K562_DNA'
H2K_COL = 'log2FoldChange_H2K'

figure_dir = 'figures'

TITLE_FSIZE = 12
AXIS_FSIZE = 12
TEXT_FSIZE = 10

greysBig = matplotlib.cm.get_cmap('Greys', 512)
greys_trunc_cm = matplotlib.colors.ListedColormap(greysBig(np.linspace(0.6, 1, 256)))

bluesBig = matplotlib.cm.get_cmap('Blues', 512)
# blues_trunc_cm = matplotlib.colors.ListedColormap(bluesBig(np.linspace(0.15, 0.8, 256)))
blues_trunc_cm = matplotlib.colors.ListedColormap(bluesBig(np.linspace(0.25, 0.8, 256)))

orangesBig = matplotlib.cm.get_cmap('Oranges', 512)
# oranges_trunc_cm = matplotlib.colors.ListedColormap(orangesBig(np.linspace(0.15, 0.8, 256)))
oranges_trunc_cm = matplotlib.colors.ListedColormap(orangesBig(np.linspace(0.25, 0.8, 256)))

round_custom_color_vec = ["#f2f3ae","#edd382","#fc9e4f","#f4442e","#020122"]

round_palette = {
    'R0': round_custom_color_vec[0],
    'R1': round_custom_color_vec[2],
    'R2': round_custom_color_vec[3],
}

cluster_name_dict = {
    'cluster_50' : 'TP53',
    'cluster_4'  : 'HNF4A/NR2F1',
    'cluster_2'  : 'HNF4A/HNF4G',
    'cluster_5'  : 'TEF',
    'cluster_8'  : 'SOX4/SOX11',
    'cluster_39' : 'NR4A1',
    'cluster_112': 'ZNF816',
    'cluster_9'  : 'SPIB/ELK1',
    'cluster_1'  : 'NFE2/JUNB',
    'cluster_10' : 'FOXP2/FOXD2',
    'cluster_11' : 'GATA2/GATA5',
    'cluster_43' : 'STAT5A',
    'cluster_120': 'GATA1::TAL1',
    'cluster_32' : 'CTCFL',
    'cluster_27' : 'SP/KLF',
    'cluster_76' : 'CTCF',
    'cluster_65' : 'ZNF460',
    'cluster_78' : 'ZNF320',
    'cluster_135': 'RREB1',
    'cluster_62' : 'CTCFL',
    'cluster_41' : 'HNF1A',
    'cluster_7'  : 'TFAP2B',
    'cluster_14' : 'NR1H4::RXRA',
    'cluster_37' : 'SCRT2',
    'cluster_81' : 'ZNF652',
    'cluster_6'  : 'BHLHA15',
    'cluster_79' : 'ZFX',
    'cluster_13' : 'TBX6',
    'cluster_58' : 'THRA',
    'cluster_42' : 'STAT2',
    'cluster_47' : 'NR2F6',
    'cluster_53' : 'E2F6',
    'cluster_99' : 'ZNF530',
    'cluster_18' : 'CREB1',
    'cluster_90' : 'EWSR1-FLI1',
    'cluster_21' : 'USF1',
    'cluster_100': 'ZNF701',
    'cluster_40' : 'TFAP4::ETV1',
    'cluster_56' : 'ZIC2',
    'cluster_62' : 'PLAGL2',
    'cluster_49' : 'ZNF416',
    'cluster_96' : 'ERF::NHLH1',
    'cluster_35' : 'PAX3',
    'cluster_17' : 'NFYB',
    'cluster_67': 'NRF1',
    'cluster_69': 'ZBTB33',
    'cluster_45': 'ZNF384'
}

def seq_to_one_hot(seq,order_dict = {'A':0, 'T':3, 'C':1, 'G':2}):
    x = np.zeros((len(seq), 4))
    for (i, bp) in enumerate(seq):
        x[i, order_dict[bp]] = 1
    return x

def one_hot_to_seq(one_hot, rev_order_dict = {0:'A', 3:'T', 1:'C', 2:'G'}):
    seq = ''
    for i in range(one_hot.shape[0]):
        seq += rev_order_dict[np.argmax(one_hot[i])]
    return seq

# add ax argument
def plot_seq_with_motifs(seq,motifs,ax=None,by_cluster=False):
    colors = {0:'#0f9447ff',1:'#235c99ff',2:'#f5b328ff',3:'#d42638ff'}

    len_per_bp = 50/145
    width = len(seq)*len_per_bp

    if ax is None:
        fig = plt.figure(figsize=(width,1))
        ax = fig.add_subplot(111)

    # fig = plt.figure(figsize=(width,1))
    # ax = fig.add_subplot(111)
    onehot = seq_to_one_hot(seq)

    viz_sequence.plot_weights_given_ax(ax,onehot, height_padding_factor=0.2,length_padding=1,subticks_frequency=20,highlight={},colors=colors)
    # drop spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # remove y ticks
    ax.set_yticks([])

   # add a horizonal line at a height of 0.9 * upper ylim from start to stop of each row in motifs

    # increase upper ylim to make room for horizontal lines
    ax.set_ylim(0,1.1*ax.get_ylim()[1])
    ann_height_prop = 0.9
    text_height_prop = 1.1
    bar_height = 0.1
    for i in range(motifs.shape[0]):
        ax.hlines(ann_height_prop*ax.get_ylim()[1],motifs.iloc[i]['start'],motifs.iloc[i]['stop'],linewidth=3,color='k')
        # add small vertical lines at start and stop of each row in motifs, centered at 0.9*upper ylim
        ax.vlines(motifs.iloc[i]['start'],ann_height_prop*ax.get_ylim()[1]-bar_height*ax.get_ylim()[1],ann_height_prop*ax.get_ylim()[1]+bar_height*ax.get_ylim()[1],linewidth=3,color='k')
        ax.vlines(motifs.iloc[i]['stop'],ann_height_prop*ax.get_ylim()[1]-bar_height*ax.get_ylim()[1],ann_height_prop*ax.get_ylim()[1]+bar_height*ax.get_ylim()[1],linewidth=3,color='k')
        # add the text of motif_alt_id above line, centered
        if by_cluster:
            ax.text((motifs.iloc[i]['start']+motifs.iloc[i]['stop'])/2,text_height_prop*ax.get_ylim()[1],motifs.iloc[i]['jaspar_cluster'],ha='center',va='center',fontsize=14)
        else:
            ax.text((motifs.iloc[i]['start']+motifs.iloc[i]['stop'])/2,text_height_prop*ax.get_ylim()[1],motifs.iloc[i]['motif_alt_id'],ha='center',va='center',fontsize=14)

    if ax is None:
        plt.show()

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model, load_model
# import Average
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, GlobalMaxPooling1D, concatenate, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute
from tensorflow.keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply, Average
import tensorflow.keras.backend as K
def load_model_for_shap(model_dir,model_basename,model_idx,model_suffix=''):

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

def load_ensemble_model(model_dir,model_basename,model_inds,model_suffix=''):

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

import scipy
def plot_scatter_shaded(x, y, ax, xlim=None, ylim=None, linreg=False, cmap=None, label=None,alpha=1,s=20):

    if cmap is None:
        greysBig = matplotlib.cm.get_cmap('Greys', 512)
        greys_trunc_cm = matplotlib.colors.ListedColormap(greysBig(np.linspace(0.6, 1, 256)))
        cmap = greys_trunc_cm
    
    xy = np.vstack([x, y])
    z = scipy.stats.gaussian_kde(xy)(xy)
    ax.scatter(
        x,
        y,
        alpha=alpha,
        s = s,
        c=z,
        cmap=cmap,
        rasterized=True,
        label=label,
        zorder=10
    )

    if linreg:
        lrres = scipy.stats.linregress(
            x,
            y,
        )
        ax.axline((0, lrres.intercept), slope=lrres.slope, color='dodgerblue', linewidth=2)
        if xlim is None:
            xlim = ax.get_xlim()
        else:
            ax.set_xlim(xlim)
        if ylim is None:
            ylim = ax.get_ylim()
        else:
            ax.set_ylim(ylim)
        ax.annotate(
            f'$r^2$ = {lrres.rvalue**2:.3f}',
            xy=(xlim[0], ylim[1]),
            xytext=(4,-4), textcoords='offset points',
            va='top',
        )

        return lrres