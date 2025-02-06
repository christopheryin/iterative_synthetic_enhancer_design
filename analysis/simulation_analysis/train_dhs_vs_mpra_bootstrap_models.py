import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf

from scipy.stats import pearsonr,spearmanr

import tensorflow.keras
from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, GlobalMaxPooling1D, concatenate, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute
from tensorflow.keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import tensorflow.keras.losses

from sklearn.model_selection import train_test_split

HEPG2_COL = 'log2FoldChange_HepG2'
K562_COL = 'log2FoldChange_K562'

def seq_to_one_hot(seq,order_dict = {'A':0, 'T':3, 'C':1, 'G':2}):
    x = np.zeros((len(seq), 4))
    for (i, bp) in enumerate(seq):
        x[i, order_dict[bp]] = 1
    return x

def seq_to_one_hot_and_pad(seq,pad_len=200,order_dict = {'A':0, 'T':3, 'C':1, 'G':2}):
    x = np.zeros((len(seq), 4))
    for (i, bp) in enumerate(seq):
        x[i, order_dict[bp]] = 1
    seq_len = len(seq)
    if seq_len < pad_len:
        lpad = (pad_len-seq_len)//2
        rpad = pad_len - seq_len - lpad
        x = np.pad(x,((lpad,rpad),(0,0)),'constant')
    return x

def extract_val_train(hepg2_df, k562_df, n_val_hepg2, n_val_k562,
                      rand_seed,HEPG2_COL=HEPG2_COL, K562_COL=K562_COL):
    
    # shuffle the dataframes - frac = 100%, replace = False by default so yeah this works
    hepg2_df = hepg2_df.sample(frac=1,random_state=rand_seed).reset_index(drop=True)
    k562_df = k562_df.sample(frac=1,random_state=rand_seed).reset_index(drop=True)

    # extract the val_ratio of the data for validation
    x_valid_hepg2 = np.stack(hepg2_df.loc[:n_val_hepg2,'enhancer'].apply(lambda x: seq_to_one_hot_and_pad(x)).values)
    x_valid_k562 = np.stack(k562_df.loc[:n_val_k562,'enhancer'].apply(lambda x: seq_to_one_hot_and_pad(x)).values)
    x_valid = np.concatenate([x_valid_hepg2,x_valid_k562])

    y_valid_hepg2 = hepg2_df.loc[:n_val_hepg2,[HEPG2_COL,K562_COL]].values
    y_valid_k562 = k562_df.loc[:n_val_k562,[HEPG2_COL,K562_COL]].values
    y_valid = np.concatenate([y_valid_hepg2,y_valid_k562])

    hepg2_train_df = hepg2_df.iloc[n_val_hepg2:].reset_index(drop=True)
    k562_train_df = k562_df.iloc[n_val_k562:].reset_index(drop=True)
    return x_valid, y_valid, hepg2_train_df, k562_train_df

def extract_bootstraps(hepg2_df, k562_df, n, rng):
    # randomly sample n//2 sequences from each cell type
    # if n is odd, sample one extra sequence from one of the cell types
    if n % 2 == 1:
        n+=1
    hepg2_seq_inds = rng.choice(hepg2_df.index,n//2,replace=True)
    k562_seq_inds = rng.choice(k562_df.index,n//2,replace=True)
    x_hepg2 = np.stack(hepg2_df.loc[hepg2_seq_inds,'enhancer'].apply(lambda x: seq_to_one_hot_and_pad(x)).values)
    x_k562 = np.stack(k562_df.loc[k562_seq_inds,'enhancer'].apply(lambda x: seq_to_one_hot_and_pad(x)).values)
    x_tot = np.concatenate([x_hepg2,x_k562])
    # x_tot = np.pad(x_tot, ((0,0),(lpad,rpad),(0,0)), 'constant', constant_values=0)

    y_hepg2 = hepg2_df.loc[hepg2_seq_inds,[HEPG2_COL,K562_COL]].values
    y_k562 = k562_df.loc[k562_seq_inds,[HEPG2_COL,K562_COL]].values
    y_tot = np.concatenate([y_hepg2,y_k562])

    return x_tot, y_tot


### Define model and training hyperparameters ###
n_epochs = 100
batch_size = 64 

cf_dir = 'r0_r1_revision_splits' # this is repo where d3 crossfold data is stored # just rerun this script changing this cf_dir location
cf_model_dir = '../shap_extraction/model_dir/M0' # repo where models to finetune are stored
results_basename = f'dhs_vs_mpra_revision_results'
# directory to save finetuned models, results
save_dir = f'dhs_vs_mpra_ft_models'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Load data from dataframes
d2_deseq_df = pd.read_csv(f'd2_deseq_df.csv')
dhs_deseq_df = pd.read_csv(f'dhs_deseq_df.csv')

d2_excluded_models = ['df','sharpr-mpra']
d2_analysis_df = d2_deseq_df[~d2_deseq_df['model_type'].isin(d2_excluded_models)].copy()
# reset index, inplace
d2_analysis_df.reset_index(inplace=True)

d2_hepg2_df = d2_analysis_df[d2_analysis_df['cell_type']=='HEPG2'].copy()
d2_k562_df = d2_analysis_df[d2_analysis_df['cell_type']=='K562'].copy()
# reset index, inplace
d2_hepg2_df.reset_index(inplace=True)
d2_k562_df.reset_index(inplace=True)

dhs_hepg2_df = dhs_deseq_df[dhs_deseq_df['cell_type']=='HEPG2'].copy()
dhs_k562_df = dhs_deseq_df[dhs_deseq_df['cell_type']=='K562'].copy()
# reset index, inplace
dhs_hepg2_df.reset_index(inplace=True)
dhs_k562_df.reset_index(inplace=True)

n_tot_seqs = 1000
n_models = 9
val_ratio = 0.15

n_hepg2_d2 = d2_hepg2_df.shape[0]
n_k562_d2 = d2_k562_df.shape[0]
n_hepg2_dhs = dhs_hepg2_df.shape[0]
n_k562_dhs = dhs_k562_df.shape[0]

n_val_hepg2_d2 = int(val_ratio*n_hepg2_d2)
n_val_k562_d2 = int(val_ratio*n_k562_d2)
n_val_hepg2_dhs = int(val_ratio*n_hepg2_dhs)
n_val_k562_dhs = int(val_ratio*n_k562_dhs)

# not used now but could be used if total number of sequences used in training should account for number of sequences in validation set
n_val_d2 = n_val_hepg2_d2 + n_val_k562_d2
n_val_dhs = n_val_hepg2_dhs + n_val_k562_dhs

print(f'Number of sequences in validation set: {n_val_d2} (D2), {n_val_dhs} (DHS)')

# calculate padding parameters - note I will only need to pad the r1-mpra data
seq_len = 145
pad_len = 200
lpad = (pad_len-seq_len)//2
rpad = pad_len - seq_len - lpad

# R2 test set - will be all 145bp so needs to be padded to 200bp

x_test = np.load(f'{cf_dir}/x_test.npy')
y_test = np.load(f'{cf_dir}/y_test.npy')

# pad the x_test with all 0s by lpad on the left, rpad on the right (axis 1)
x_test = np.pad(x_test, ((0,0),(lpad,rpad),(0,0)), 'constant', constant_values=0)

n_dhs_seqs = [n_tot_seqs,n_tot_seqs // 2, 0]

for cur_n_dhs in n_dhs_seqs:

    cur_n_r1 = n_tot_seqs - cur_n_dhs

    # one save_dir per ratio, will have a corresponding results file for all 9 models that get trained
    cur_save_dir = f'{save_dir}/DvM_{cur_n_dhs}-{cur_n_r1}'
    if not os.path.isdir(cur_save_dir):
        os.makedirs(cur_save_dir)

    ft_suffix = f'_DvM_{cur_n_dhs}-{cur_n_r1}_v'

    # for each model, bootstrap a new training set and finetune
    for model_idx in range(1,10):

        # load the model
        K.clear_session() # frees up memory, otherwise model training slows down a lot when training models in a loop
        model_basename = f'd1_wide_cf_t0_v{model_idx}'
        model_name = f'{cf_model_dir}/{model_basename}.h5'
        model = load_model(model_name)

        # generate the bootstrap according to the current ratio

        # randomly sample n_r0_seqs from dhs_deseq_df, with replacement
        # first set random seed
        rand_seed = cur_n_dhs + model_idx
        rng = np.random.default_rng(rand_seed)

        # randomly shuffle the dfs inplace, extract held out validation set, and training set which will be bootstrapped
        # (don't want to pull validation set from bootstrapped set because of repeat sequences from bootstrap)
        # validation sequences will be padded to 200bp in this function
        # train data extracted as dfs so can be piped into the extract_bootstraps function
        x_valid_d2, y_valid_d2,d2_hepg2_train_df, d2_k562_train_df = extract_val_train(d2_hepg2_df, d2_k562_df, n_val_hepg2_d2, n_val_k562_d2, rand_seed)
        x_valid_dhs, y_valid_dhs,dhs_hepg2_train_df, dhs_k562_train_df = extract_val_train(dhs_hepg2_df, dhs_k562_df, n_val_hepg2_dhs, n_val_k562_dhs, rand_seed)

        # combine validation arrays
        x_valid = np.concatenate([x_valid_d2,x_valid_dhs])
        y_valid = np.concatenate([y_valid_d2,y_valid_dhs])

        # extract sequences from dhs_deseq_df and d2_analysis_df
        # mpra only
        if cur_n_dhs == 0:
            # bootstrap n_tot_seqs//2 from each cell type
            x_tot,y_tot = extract_bootstraps(d2_hepg2_train_df, d2_k562_train_df, cur_n_r1, rng)
        # dhs only
        elif cur_n_r1 == 0:
            # bootstrap n_tot_seqs//2
            x_tot,y_tot = extract_bootstraps(dhs_hepg2_train_df, dhs_k562_train_df, cur_n_dhs, rng)
        # 50:50 mix
        else:
            x_mpra, y_mpra = extract_bootstraps(d2_hepg2_train_df, d2_k562_train_df, cur_n_r1//2, rng)
            x_dhs, y_dhs = extract_bootstraps(dhs_hepg2_train_df, dhs_k562_train_df, cur_n_dhs//2, rng)
            x_tot = np.concatenate([x_mpra,x_dhs])
            y_tot = np.concatenate([y_mpra,y_dhs])

        # get 25th and 75th percentile of y_tot[:,0]-y_tot[:,1], save to text file
        with open(f'{cur_save_dir}/{results_basename}{ft_suffix[:-1]}percentiles.txt', 'a') as f:
            y_h2k = y_tot[:,0]-y_tot[:,1]
            f.write(f'{np.percentile(y_h2k,25):.3f},{np.percentile(y_h2k,75):.3f}\n')


        # note this means de fact the amount of sequences I'm training with is n_val + n_train, not 1000. I could reduce the n if I wanted to to compensate accordingly but I won't for now
        x_train = x_tot
        y_train = y_tot

        # add reverse complements to the training set
        x_train = np.concatenate([
            x_train,
            x_train[:, ::-1, ::-1]
        ], axis=0)

        y_train = np.concatenate([
            y_train,
            y_train
        ], axis=0)


        # not going to bother retuning these in the interest of not rabbit-holing
        lr = 1e-4
        factor = 0.25
        min_delta_r = 1e-2
        min_delta_se = 1e-3
        patience_r = 4
        patience_se = 5

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,beta_1=0.9,beta_2=0.999), loss='mse')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, min_delta=min_delta_r,
                                    patience=patience_r, min_lr=lr*1e-3)
        stop_early = EarlyStopping(monitor='val_loss', min_delta = min_delta_se, patience=patience_se,restore_best_weights=True)
        callbacks = [reduce_lr, stop_early]

        history = model.fit(x_train, y_train, batch_size=64, epochs=250, validation_data=(x_valid, y_valid), callbacks=callbacks)

        model_path = os.path.join(cur_save_dir, f'{model_basename}{ft_suffix[:-2]}.h5')
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        # calculate metrics
        y_test_hat = model.predict(x_test)

        metrics = ['pearsonr','spearmanr']
        for corr_idx,corr in enumerate([pearsonr,spearmanr]):
            r_h_d2   = corr(y_test[:,0],y_test_hat[:,0])[0]
            r_k_d2   = corr(y_test[:,1],y_test_hat[:,1])[0]
            r_h2k_d2   = corr(y_test[:,0]-y_test[:,1],y_test_hat[:,0]-y_test_hat[:,1])[0]
                
            # print results to text file
            with open(f'{cur_save_dir}/{results_basename}{ft_suffix[:-1]}{metrics[corr_idx]}.txt', 'a') as f:
                # write the hyperparameters for this model
                f.write(f'{r_h_d2:.3f},{r_k_d2:.3f},{r_h2k_d2:.3f}\n')