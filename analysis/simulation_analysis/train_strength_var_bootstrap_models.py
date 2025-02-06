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
H2K_COL = 'log2FoldChange_H2K'

def seq_to_one_hot(seq,order_dict = {'A':0, 'T':3, 'C':1, 'G':2}):
    x = np.zeros((len(seq), 4))
    for (i, bp) in enumerate(seq):
        x[i, order_dict[bp]] = 1
    return x

### Define model and training hyperparameters ###
n_epochs = 100
batch_size = 64 

cf_dir = 'r0_r1_revision_splits' # this is repo where d3 crossfold data is stored # just rerun this script changing this cf_dir location
cf_model_dir = '../shap_extraction/model_dir/M0' # repo where models to finetune are stored
results_basename = f'rand_vs_high_results'
# directory to save finetuned models, results
save_dir = f'rand_vs_high_ft_models'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Load data from dataframes
d2_deseq_df = pd.read_csv(f'd2_deseq_df.csv')
d1_deseq_plus_cluster_cnts_df = pd.read_csv(f'd1_deseq_plus_cluster_cnts_df.csv')

d2_excluded_models = ['df','sharpr-mpra']
d2_analysis_df = d2_deseq_df[~d2_deseq_df['model_type'].isin(d2_excluded_models)].copy()
d2_analysis_df.reset_index(inplace=True)

# sort by H2K_COL
d2_analysis_df.sort_values(H2K_COL,inplace=True)

# partition into 4 groups
n_seqs = d2_analysis_df.shape[0]
n_quartiles = n_seqs//4
quartile_inds = np.arange(0,n_seqs,n_quartiles)
quartile_inds[-1] = n_seqs

df0 = d2_analysis_df.iloc[quartile_inds[0]:quartile_inds[1]].copy()
df1 = d2_analysis_df.iloc[quartile_inds[1]:quartile_inds[2]].copy()
df2 = d2_analysis_df.iloc[quartile_inds[2]:quartile_inds[3]].copy()
df3 = d2_analysis_df.iloc[quartile_inds[3]:].copy()

# now let's create 3 different training sets: 1) only weak, 2) only strong, 3) all

df_weak = pd.concat([df1,df2],ignore_index=True)
df_strong = pd.concat([df0,df3],ignore_index=True)
df_all = d2_analysis_df.copy()

# reset index for each
df_weak.reset_index(inplace=True,drop=True)
df_strong.reset_index(inplace=True,drop=True)

dfs = [df_weak,df_strong,df_all]
df_names = ['weak','strong','all']

n_tot_seqs = 1000
n_models = 9 # let's just finetune each model once

# calculate padding parameters
seq_len = 145
pad_len = 200
lpad = (pad_len-seq_len)//2
rpad = pad_len - seq_len - lpad

# Load R2 test set

x_test = np.load(f'{cf_dir}/x_test.npy')
y_test = np.load(f'{cf_dir}/y_test.npy')

# pad the x_test with all 0s by lpad on the left, rpad on the right (axis 1)
x_test = np.pad(x_test, ((0,0),(lpad,rpad),(0,0)), 'constant', constant_values=0)

# first set random seed
rand_seed = 2048
rng = np.random.default_rng(rand_seed)

for df,df_name,seed_idx in zip(dfs,df_names,np.arange(len(dfs))):

    # one save_dir per ratio, will have a corresponding results file for all 9 models that get trained
    cur_save_dir = f'{save_dir}/{df_name}'
    os.makedirs(cur_save_dir,exist_ok=True)

    ft_suffix = df_name

    # for each model, bootstrap a new training set and finetune
    for model_idx in range(1,10):

        # load the model
        K.clear_session() # frees up memory, otherwise model training slows down a lot when training models in a loop
        model_basename = f'd1_wide_cf_t0_v{model_idx}'
        model_name = f'{cf_model_dir}/{model_basename}.h5'
        model = load_model(model_name)

        # generate the bootstrap according to the current ratio

        # randomly sample n_tot_seqs from d1_analysis_df, with replacement

        boot_inds = rng.choice(df.index,n_tot_seqs,replace=True)
        x_tot = np.stack(df.loc[boot_inds,'enhancer'].apply(lambda x: seq_to_one_hot(x)).values)
        y_tot = df.loc[boot_inds,[HEPG2_COL,K562_COL]].values

        # pad x_tot
        x_tot = np.pad(x_tot, ((0,0),(lpad,rpad),(0,0)), 'constant', constant_values=0)

        # get 25th and 75th percentile of y_tot[:,0]-y_tot[:,1], save to text file
        with open(f'{cur_save_dir}/{results_basename}{ft_suffix}percentiles.txt', 'a') as f:
            y_h2k = y_tot[:,0]-y_tot[:,1]
            f.write(f'{np.percentile(y_h2k,25):.3f},{np.percentile(y_h2k,75):.3f}\n')

        # now randomly split into 850 train, 150 valid using sklearn train_test_split
        x_train, x_valid, y_train, y_valid = train_test_split(x_tot,y_tot,test_size=0.15,random_state=model_idx+seed_idx*10)

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

        model_path = os.path.join(cur_save_dir, f'{model_basename}{ft_suffix}.h5')
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
            with open(f'{cur_save_dir}/{results_basename}{ft_suffix}{metrics[corr_idx]}.txt', 'a') as f:
                # write the hyperparameters for this model
                f.write(f'{r_h_d2:.3f},{r_k_d2:.3f},{r_h2k_d2:.3f}\n')