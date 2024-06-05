import pandas as pd
import os
import numpy as np
import seaborn as sns

from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import matplotlib
import scipy

import logomaker

# import isolearn
import itertools

import collections

from Levenshtein import distance

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, GlobalMaxPooling1D, concatenate, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute
from tensorflow.keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply, Average
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import tensorflow.keras.losses

matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
TEXT_FSIZE =14
TITLE_FSIZE = 16
AXIS_FSIZE = 14

### Define relevant paths ###

# Data directories
aws_dir = '../../aws/for_aws'

d1_data_dir = f'{aws_dir}/sharpr_mpra_data'
d2_proc_dir = f'{aws_dir}/sequencing_analysis'
d1_deseq_data_dir = f'{d2_proc_dir}/sharpr-mpra_deseq_processing/data_sharpr'
d2_deseq_data_dir = f'{d2_proc_dir}/d2_deseq_data'
merged_data_dir = f'{d2_proc_dir}/sharpr-mpra_deseq_processing/merged_data'

shap_dir = f'{d2_proc_dir}/interpretation/shap_analysis/extracted_shap_values'

# model directories - original
predictor_dir = f'{aws_dir}/predictor_models'
wide_predictor_dir = f'{predictor_dir}/wide_predictors'
boot_predictor_dir = f'{predictor_dir}/boot_predictors'

# model directories - retrained deseq d1
d1_deseq_predictor_dir = f'{aws_dir}/sequencing_analysis/sharpr-mpra_deseq_processing/d1_trained_models'
d1_deseq_model_basename = 'wide_deseq_v2_d1'

# model directories - retrained d1+d2 multidata models
multidata_predictor_dir = f'{aws_dir}/sequencing_analysis/retraining/retrained_models/re_deseq_retrained_model'
l2_model_dir            = f'{aws_dir}/sequencing_analysis/retraining/retrained_models/re_deseq_retrained_model/l2_reg'
l2_model_basename       = 'retrained_deseq_v2_multidataset_l20.001'

# motif directories
motif_dir = f'{aws_dir}/sequencing_analysis/fimo_motif_scanning/saved_processed_motif_files'

# shap directories
shap_dir = f'{aws_dir}/sequencing_analysis/interpretation/shap_analysis/extracted_shap_values'

### Cluster to representative TF name mapping ###
cluster_name_dict = {
    'cluster_50' : 'TP53',
    'cluster_4'  : 'HNF4A/NR2F1',
    'cluster_2'  : 'HNF4A/HNF4G',
    'cluster_5'  : 'TEF',
    'cluster_8'  : 'SOX4/SOX11',
    'cluster_39' : 'NR4A1',
    'cluster_112': 'ZNF816',
    'cluster_9'  : 'SPIB/EKL1',
    'cluster_1'  : 'NFE2/JUNB',
    'cluster_10' : 'FOXP2/FOXD2',
    'cluster_11' : 'GATA2/GATA5',
    'cluster_43' : 'STAT5A',
    'cluster_120': 'GATA1::TAL1',
    'cluster_32' : 'MZF1',
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
    'cluster_18' : 'CREB1',
    'cluster_68' : 'ZNF85',
    'cluster_56' : 'ZIC2',
    'cluster_90' : 'EWSR1-FLI1'
}
###

### Load data functions ###

# ugh wait I need the dna_minp_count too - well, maybe I don't? yeah!
def load_d1_og():

    x_train = np.load(f'{d1_data_dir}/sharpr_cached_minp_hepg2_k562minlogfold-2.4_minDNA200_x_train.npy')
    y_train = np.load(f'{d1_data_dir}/sharpr_cached_minp_hepg2_k562minlogfold-2.4_minDNA200_y_train.npy')
    x_valid = np.load(f'{d1_data_dir}/sharpr_cached_minp_hepg2_k562minlogfold-2.4_minDNA200_x_valid.npy')
    y_valid = np.load(f'{d1_data_dir}/sharpr_cached_minp_hepg2_k562minlogfold-2.4_minDNA200_y_valid.npy')
    x_test = np.load(f'{d1_data_dir}/sharpr_cached_minp_hepg2_k562minlogfold-2.4_minDNA200_x_test.npy')
    y_test = np.load(f'{d1_data_dir}/sharpr_cached_minp_hepg2_k562minlogfold-2.4_minDNA200_y_test.npy')

    x_train = x_train[:, 0, :, :]
    x_valid = x_valid[:, 0, :, :]
    x_test = x_test[:, 0, :, :]

    n_train = x_train.shape[0] // 2
    n_valid = x_valid.shape[0] // 2
    n_test = x_test.shape[0] // 2

    # remove RCs
    x_train = x_train[:n_train]
    x_valid = x_valid[:n_valid]
    x_test = x_test[:n_test]
    y_train = y_train[:n_train]
    y_valid = y_valid[:n_valid]
    y_test = y_test[:n_test]

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def load_d1_deseq():

    x_train = np.load(f'{d1_deseq_data_dir}/x_d1_train.npy')
    y_train = np.load(f'{d1_deseq_data_dir}/y_d1_train.npy')
    x_valid = np.load(f'{d1_deseq_data_dir}/x_d1_valid.npy')
    y_valid = np.load(f'{d1_deseq_data_dir}/y_d1_valid.npy')
    x_test = np.load(f'{d1_deseq_data_dir}/x_d1_test.npy')
    y_test = np.load(f'{d1_deseq_data_dir}/y_d1_test.npy')

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def load_d1_deseq_df():
    return pd.read_csv(f'{d1_deseq_data_dir}/final_deseq_sharpr_filt_no_dups_df.csv')

def load_d2_deseq():
    x_train = np.load(f'{d2_deseq_data_dir}/x_d2_train.npy')
    y_train = np.load(f'{d2_deseq_data_dir}/y_d2_train.npy')
    x_valid = np.load(f'{d2_deseq_data_dir}/x_d2_valid.npy')
    y_valid = np.load(f'{d2_deseq_data_dir}/y_d2_valid.npy')
    x_test = np.load(f'{d2_deseq_data_dir}/x_d2_test.npy')
    y_test = np.load(f'{d2_deseq_data_dir}/y_d2_test.npy')

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def load_d2_deseq_df():

    log2fc_df = pd.read_csv(f'{d2_deseq_data_dir}/chris_log2fc_df_clean2.csv')

    # rename cell types to K562, HEPG2, K562 Control, HEPG2 Control
    log2fc_df.loc[log2fc_df['cell_type']=='k562','cell_type'] = 'K562'
    log2fc_df.loc[log2fc_df['cell_type']=='hepg2','cell_type'] = 'HEPG2'
    log2fc_df.loc[log2fc_df['cell_type']=='k562_ctrl','cell_type'] = 'K562 Control'
    log2fc_df.loc[log2fc_df['cell_type']=='hepg2_ctrl','cell_type'] = 'HEPG2 Control'

    # For all HEPG2 Control and K562 control sequences, change model name to control_f if "f_" is in the name, and control_r if "r_" is in the name
    log2fc_df.loc[log2fc_df['cell_type']=="HEPG2 Control",'model'] = log2fc_df.loc[log2fc_df['cell_type']=="HEPG2 Control",'model'].apply(lambda x: "control_f" if "f_" in x else "control_r")
    log2fc_df.loc[log2fc_df['cell_type']=="K562 Control",'model'] = log2fc_df.loc[log2fc_df['cell_type']=="K562 Control",'model'].apply(lambda x: "control_f" if "f_" in x else "control_r")
    # now change cell_type to HEPG2 if HEPG2 Control, and K562 if K562 Control
    log2fc_df.loc[log2fc_df['cell_type']=="HEPG2 Control",'cell_type'] = "HEPG2"
    log2fc_df.loc[log2fc_df['cell_type']=="K562 Control",'cell_type'] = "K562"

    # if d2_deseq_df['model'] contains crafted, change 'generator' to 'motif_repeat'
    log2fc_df.loc[log2fc_df['model'].str.contains("crafted"),'generator'] = "motif_repeat"

    return log2fc_df

def load_merged_d1_d2():

    x_train = np.load(f'{merged_data_dir}/x_merged_train.npy')
    y_train = np.load(f'{merged_data_dir}/y_merged_train.npy')
    w_train = np.load(f'{merged_data_dir}/w_merged_train.npy')
    x_valid = np.load(f'{merged_data_dir}/x_merged_valid.npy')
    y_valid = np.load(f'{merged_data_dir}/y_merged_valid.npy')
    w_valid = np.load(f'{merged_data_dir}/w_merged_valid.npy')
    x_test = np.load(f'{merged_data_dir}/x_merged_test.npy')
    y_test = np.load(f'{merged_data_dir}/y_merged_test.npy')
    w_test = np.load(f'{merged_data_dir}/w_merged_test.npy')

    print('W order: HEPG2 D1, HEPG2 D2, K562 D1, K562 D2')

    return x_train, x_valid, x_test, y_train, y_valid, y_test, w_train, w_valid, w_test

def load_motif_data_d1(qthresh_suffix=''):
    d1_final_df = pd.read_csv(f'{motif_dir}/d1_final_df{qthresh_suffix}_v2.csv')
    d1_clustered_motif_df = pd.read_csv(f'{motif_dir}/d1_clustered_motif_df{qthresh_suffix}.csv')
    d1_deseq_plus_cluster_cnts_df = pd.read_csv(f'{motif_dir}/d1_deseq_plus_cluster_cnts_df{qthresh_suffix}_v2.csv')

    return d1_final_df, d1_clustered_motif_df, d1_deseq_plus_cluster_cnts_df

def load_motif_data_d2(qthresh_suffix='_qthresh05'):
    d2_final_df = pd.read_csv(f'{motif_dir}/d2_final_df{qthresh_suffix}_v2.csv')
    d2_clustered_motif_df = pd.read_csv(f'{motif_dir}/d2_clustered_motif_df{qthresh_suffix}.csv')
    d2_deseq_plus_cluster_cnts_df = pd.read_csv(f'{motif_dir}/d2_deseq_plus_cluster_cnts_df{qthresh_suffix}_v2.csv')


    # For all HEPG2 Control and K562 control sequences, change model name to control_f if "f_" is in the name, and control_r if "r_" is in the name
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="HEPG2_CTRL",'model'] = d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="HEPG2_CTRL",'model'].apply(lambda x: "control_f" if "f_" in x else "control_r")
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="K562_CTRL",'model'] = d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="K562_CTRL",'model'].apply(lambda x: "control_f" if "f_" in x else "control_r")
    # now change cell_type to HEPG2 if HEPG2 Control, and K562 if K562 Control
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="HEPG2_CTRL",'cell_type'] = "HEPG2"
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="K562_CTRL",'cell_type'] = "K562"

    # if d2_deseq_df['model'] contains crafted, change 'generator' to 'motif_repeat'
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['model'].str.contains("crafted"),'generator'] = "motif_repeat"


    return d2_final_df, d2_clustered_motif_df, d2_deseq_plus_cluster_cnts_df

def load_motif_data_dw(qthresh_suffix=''):
    d2_final_df = pd.read_csv(f'{motif_dir}/dw_final_df{qthresh_suffix}_v2.csv')
    d2_clustered_motif_df = pd.read_csv(f'{motif_dir}/dw_clustered_motif_df{qthresh_suffix}.csv')
    d2_deseq_plus_cluster_cnts_df = pd.read_csv(f'{motif_dir}/dw_deseq_plus_cluster_cnts_df{qthresh_suffix}_v2.csv')

    return d2_final_df, d2_clustered_motif_df, d2_deseq_plus_cluster_cnts_df


def load_motif_data_shendure(qthresh_suffix=''):
    ds_final_df = pd.read_csv(f'{motif_dir}/shendure_final_df{qthresh_suffix}_v2.csv')
    ds_clustered_motif_df = pd.read_csv(f'{motif_dir}/shendure_clustered_motif_df{qthresh_suffix}.csv')
    ds_deseq_plus_cluster_cnts_df = pd.read_csv(f'{motif_dir}/shendure_multicell_plus_cluster_cnts_df{qthresh_suffix}_v2.csv')

    return ds_final_df, ds_clustered_motif_df, ds_deseq_plus_cluster_cnts_df

# outdated, can remove now that load_motif_data_d2 accepts optional qthresh_suffix
def load_motif_data_d2_qthresh05():
    d2_final_df = pd.read_csv(f'{motif_dir}/d2_final_df_qthresh05.csv')
    d2_clustered_motif_df = pd.read_csv(f'{motif_dir}/d2_clustered_motif_df_qthresh05.csv')
    d2_deseq_plus_cluster_cnts_df = pd.read_csv(f'{motif_dir}/d2_deseq_plus_cluster_cnts_df_qthresh05.csv')

    return d2_final_df, d2_clustered_motif_df, d2_deseq_plus_cluster_cnts_df

def load_jaspar_cluster_df():
    jaspar_cluster_df = pd.read_csv(f'{motif_dir}/../jaspar_motif_clusters.tsv',sep='\t',header=None,names=['cluster','motifs'])
    jaspar_cluster_df['motifs'] = jaspar_cluster_df['motifs'].apply(lambda x: x.upper())
    return jaspar_cluster_df


### Load model functions ###

def ensemble_prediction(model_dir,model_basename,model_inds,x_test,y_test):

    # if model_inds not an iterable, make it one
    if not isinstance(model_inds, collections.Iterable):
        model_inds = [model_inds]
    
    y_test_hat_cum = np.zeros(y_test.shape)

    for model_ind in model_inds:
        K.clear_session()
        model = load_model(f'{model_dir}/{model_basename}_{model_ind}.h5')
        y_test_hat = model.predict(x_test)
        y_test_hat_cum += y_test_hat

    y_test_hat_cum /= len(model_inds)
    print(f'Number of models ensembled: {len(model_inds)}')

    return y_test_hat_cum

def load_ensemble_model(model_dir,model_basename,model_inds):

    assert isinstance(model_inds, collections.Iterable), "model_inds must be a list or array"

    models = [None] * len(model_inds)
    for i in range(len(model_inds)):
        models[i] = load_model(f'{model_dir}/{model_basename}_{model_inds[i]}.h5')
        models[i]._name = f"model_idx{i}"

    ensemble_input = Input(shape=models[0].input_shape[1:])
    ensemble_outputs = [model(ensemble_input) for model in models]
    ensemble_avg = Average()(ensemble_outputs)
    ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_avg)

    return ensemble_model

# replaces GlobalMaxPooling1D with MaxPooling1D(pool_size=145), since SHAP doesn't support GlobalMaxPooling1D
def load_model_for_shap(model_dir,model_basename,model_idx):

    model = load_model(f'{model_dir}/{model_basename}_{model_idx}.h5')

    n_filters = 608
    filt_sizes = [15,11,21]
    n_dense = 288
    dropout_rate = 0.0
    n_classes = 2
    n_datasets = 2
    lr = 0.0006

    sequence_input = Input(shape=(145, 4),name="pat_input")

    convs = [None]*len(filt_sizes)

    for i in range(len(filt_sizes)):
        conv1           = Conv1D(n_filters, filt_sizes[i], padding='same', activation='linear', name = "pat_conv_" + str(i))(sequence_input)
        batchnorm1      = BatchNormalization(axis=-1,name = "pat_batchnorm_" + str(i))(conv1)
        exp1            = Activation('exponential',name = "pat_relu_" + str(i))(batchnorm1)
        convs[i]        = Dropout(dropout_rate,name = "pat_dropout_" + str(i))(MaxPooling1D(pool_size=145,name = "pat_pool_" + str(i))(exp1))
        convs[i]        = Lambda(lambda x: K.squeeze(x, axis=1),name = "pat_squeeze_" + str(i))(convs[i])

    concat1           = concatenate(convs,name="pat_concat_layer")

    dense           = Dense(n_dense,activation='relu',name="pat_dense")(concat1)
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

def load_reg_model_for_shap(model_dir,model_basename,model_idx):

    model = load_model(f'{model_dir}/{model_basename}_{model_idx}.h5')

    n_filters = 608
    filt_sizes = [15,11,21]
    n_dense = 224
    dropout_rate = 0.184
    n_classes = 2
    n_datasets = 2
    lr = 0.0003

    sequence_input = Input(shape=(145, 4),name="pat_input")

    convs = [None]*len(filt_sizes)

    for i in range(len(filt_sizes)):
        conv1           = Conv1D(n_filters, filt_sizes[i], padding='same', activation='linear', name = "pat_conv_" + str(i))(sequence_input)
        batchnorm1      = BatchNormalization(axis=-1,name = "pat_batchnorm_" + str(i))(conv1)
        exp1            = Activation('exponential',name = "pat_relu_" + str(i))(batchnorm1)
        convs[i]        = Dropout(dropout_rate,name = "pat_dropout_" + str(i))(MaxPooling1D(pool_size=145,name = "pat_pool_" + str(i))(exp1))
        convs[i]        = Lambda(lambda x: K.squeeze(x, axis=1),name = "pat_squeeze_" + str(i))(convs[i])

    concat1           = concatenate(convs,name="pat_concat_layer")

    dense           = Dense(n_dense,activation='relu',name="pat_dense")(concat1)
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

### Auxiliary functions ###

def seq_to_one_hot(seq,order_dict = {'A':0, 'T':3, 'C':1, 'G':2}):
    x = np.zeros((len(seq), 4))
    for (i, bp) in enumerate(seq):
        x[i, order_dict[bp]] = 1
    return x

def one_hot_to_seq(x):
    seq = ''
    for i in range(x.shape[0]):
        seq += 'ACGT'[np.argmax(x[i,:])]
    return seq

# function that prints all the function names defined in this module, excluding the ones imported from other modules
def print_functions():
    import inspect, sys
    print('Local functions in figure_utils.py:')
    print('----------------------------------')
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj) and obj.__module__ == __name__:
            print(name)

### Sequence logo functions ###

def plot_and_save_logo(pwm_ic,svg_save_dir,motif_name,height_per_row=0.8,width_per_col=1.5,rc=False,suffix='STREME'):
    """_summary_

    Args:
        pwm_ic (np.array): n x 4 array of PWM IC values (PWM weighted by position-wise IC) - hmm maybe I should have IC be calculated in this function? needs nsite tho
        svg_save_dir (String): _description_
        motif_name (String): _description_
        height_per_row (float, optional): Height per row (total column height). Defaults to 0.8.
        width_per_col (float, optional): Width per col (position). Defaults to 1.5.
        rc (bool, optional): When True plot reverse complement of motif. Defaults to False.
    """
    num_cols = pwm_ic.shape[0]
    num_rows = pwm_ic.shape[1]
    fig = plt.figure(figsize=[width_per_col * num_cols, height_per_row * num_rows])
    ax = fig.add_subplot(111)
    logo_colors = {'A':'#0f9447ff','C':'#235c99ff','G':'#f5b328ff','T':'#d42638ff'}
    pwm_ic_df = pd.DataFrame(pwm_ic, columns=['A','C','G','T'])
    # if reverse complement, get reverse complement of pwm_ic_df instead
    if rc:
        pwm_ic_df = pwm_ic_df.iloc[::-1].reset_index(drop=True)
        pwm_ic_df = pwm_ic_df.rename(columns={'A':'T','C':'G','G':'C','T':'A'})

    logo = logomaker.Logo(pwm_ic_df,color_scheme=logo_colors,ax=ax)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left','bottom'], visible=True)

    # set ylim to 2
    ax.set_ylim(0,2)
    # remove xticks, yticks, and all spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # save figure
    plt.savefig(f'{svg_save_dir}/{motif_name}_{suffix}.svg')
    plt.close()

    return

### FASTA and MEME read/write functions ###

def load_fasta(fname):
    names = []
    seqs = []
    with open(fname, 'r') as f:
        for line in f:
            if line[0] == '>':
                names.append(line[1:].strip())
                seqs.append('')
            else:
                seqs[-1] += line.strip()

    # combine names and seqs into a dataframe
    df = pd.DataFrame({'name': names, 'seq': seqs})

    return df

def write_to_fasta(df,fname,seq_colname='enhancer',name_colname=None):
    with open(fname,'w') as f:
        for idx, row in df.iterrows():
            if name_colname is not None:
                f.write(f'>{row[name_colname]}\n')
            else:
                f.write(f'>{idx}\n')
            f.write(f'{row[seq_colname]}\n\n')

def write_to_fasta_with_score(df,fname,seq_colname='enhancer',name_colname=None,score_colname = 'log2FoldChange_H2K_deseq'):
    with open(fname,'w') as f:
        for idx, row in df.iterrows():
            if name_colname is not None:
                f.write(f'>{row[name_colname]} {row[score_colname]}\n')
            else:
                f.write(f'>{idx} {row[score_colname]}\n')
            f.write(f'{row[seq_colname]}\n\n')

def read_pwm_from_meme(meme_dir,meme_fname):

    with open(f'{meme_dir}/{meme_fname}') as f:

        # check if end of file has been reached
        line = next(f)
        while line:
            # if line begins with 'MOTIF' then it's a new motif
            if line.startswith('MOTIF'):
                # get the motif name as the last word in the line
                motif_name = line.split()[-1]
                print(motif_name)
                # skip next line with 'letter-probability matrix: alength=  w=  nsites=  E=' annotation
                line = next(f)
                nsites = int(line.split()[7])
                # read the next lines into np array until you hit a blank line
                motif = np.array([])
                line = next(f)
                # while len(line)>1:
                while line[0:3] != 'URL':
                    motif = np.append(motif, np.array(line.split()).astype(float))
                    line = next(f)
                motif = motif.reshape(-1, 4)
            # read the next line if eof not reached
            try:
                line = next(f)
            except StopIteration:
                break

        return motif,nsites