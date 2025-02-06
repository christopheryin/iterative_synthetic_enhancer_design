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

import pickle
from sklearn.model_selection import train_test_split, KFold

### DEFINE FUNCTIONS ###

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

HEPG2_COL = 'log2FoldChange_HepG2'
K562_COL = 'log2FoldChange_K562'
H2K_COL = 'log2FoldChange_H2K'

### DEFINE MODEL AND DATA PATHS ###
cf_dir = 'cf10'
dhs_cf_dir = 'cf5_dhs'

cf_model_dir = 'M0'

### END DEFINE MODEL AND DATA PATHS ###

### LOAD IN DATA ###

# R2 test set
test_dir = '.'
x_test = np.load(f'{test_dir}/x_test.npy')
y_test = np.load(f'{test_dir}/y_test.npy')

# pad x_test to 200 bp
lpad = (200-145)//2
rpad = 200 - 145 - lpad
x_test = np.pad(x_test, ((0,0),(lpad,rpad),(0,0)), 'constant', constant_values=0)

# load all the crossfolds for d2 and dhs
cf_dir = 'cf10'

n_cfs = 10
x_cfs = [np.load(f'{cf_dir}/x_cf{i}.npy') for i in range(n_cfs)]
y_cfs = [np.load(f'{cf_dir}/y_cf{i}.npy') for i in range(n_cfs)]
w_cfs = [np.load(f'{cf_dir}/w_cf{i}.npy') for i in range(n_cfs)]

# load dhs cfs
n_dhs_cfs = 5
dhs_x_cfs = [np.load(f'{dhs_cf_dir}/dhs_x_cf{i}.npy') for i in range(n_dhs_cfs)]
dhs_y_cfs = [np.load(f'{dhs_cf_dir}/dhs_y_cf{i}.npy') for i in range(n_dhs_cfs)]
# dhs_test_cf_idx = 0

# padd to 200 bp, also extract only the d1 columns
lpad = (200-145)//2
rpad = 200 - 145 - lpad

# for each cf, extract the y_cf rows where w_cf[:,1] == 1 (these are the d2 seqs)
for cf_idx in range(n_cfs):
    x_cfs[cf_idx] = x_cfs[cf_idx][w_cfs[cf_idx][:,1] == 1]
    # pad the x_cfs with all 0s by lpad on the left, rpad on the right (axis 1)
    x_cfs[cf_idx] = np.pad(x_cfs[cf_idx], ((0,0),(lpad,rpad),(0,0)), 'constant', constant_values=0)

    y_cfs[cf_idx] = y_cfs[cf_idx][w_cfs[cf_idx][:,1] == 1]
    # extract the 0st and 2nd columns of y_cf (d2)
    y_cfs[cf_idx] = y_cfs[cf_idx][:,[1,3]]

### END LOAD IN DATA ###

# finetuning hps #
lr = 1e-4
factor = 0.25
min_delta_r = 1e-2
min_delta_se = 1e-3
patience_r = 4
patience_se = 5
# end finetuning hps #

n_bootstraps = 5 # finetune each of the 9 models on a different bootstrap (start with just 1 bootstrap)
n_ft_models = 9 # number of models to finetune per ensemble

n_vals = [100,200,300,500,1000,1350,1750][::-1]
# for each value of n, I will need to extract the n_mpra and n_train from the ratio
n_mpra_eff = int(np.sum([x_cf.shape[0] for x_cf in x_cfs]) * .9)
n_dhs_eff = int(np.sum([x_cf.shape[0] for x_cf in dhs_x_cfs]) * .8)
n_tot_eff = n_mpra_eff + n_dhs_eff
md_ratio = n_mpra_eff / n_tot_eff
# once cur_n is obtained, I will obtain the n_mpra and n_train from the ratio

rand_seed = 14
rng = np.random.default_rng(rand_seed) # set this once at the beginning of the script and then it will always be replicable

# for each value of n... 
for cur_n in n_vals:

    # initialize repo for saving models and results
    save_dir = f'nvar_model_dir_cf10_rotate_final/n{cur_n}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    results_basename = f'nvar_results_n{cur_n}'

    cur_n_mpra = int(cur_n * md_ratio)
    cur_n_dhs = cur_n - cur_n_mpra

    cur_n_mpra_train = int(cur_n_mpra * 8/9)
    cur_n_mpra_val = cur_n_mpra - cur_n_mpra_train

    print(f'cur_n_mpra: {cur_n_mpra} ({cur_n_mpra_train}/{cur_n_mpra_val}), cur_n_dhs: {cur_n_dhs}')


# ...generate 5 bootstraps...
    skip_inds = [] # this is to ensure consistent seed for bootstraps, ignore if no previous bootstraps have been generated
    for boot_idx in [0,1,2,3,4]:

        # split mpra and dhs data separately into cfs, then combine for training to exactly match original M1 training procedure
        test_cf_idx = boot_idx
        dhs_test_cf_idx = boot_idx
        model_basename = f'd1_wide_cf_t{test_cf_idx}_v'

        y_preds = []
        for val_cf_idx in [i for i in range(n_cfs) if i != test_cf_idx]:

            x_boot_valid = x_cfs[val_cf_idx]
            y_boot_valid = y_cfs[val_cf_idx]
            # downsample to cur_n_mpra_val
            if cur_n_mpra_val > x_boot_valid.shape[0]:
                print(f'cur_n_mpra_val ({cur_n_mpra_val}) > x_boot_valid.shape[0] ({x_boot_valid.shape[0]}), sampling {x_boot_valid.shape[0]}')
                cur_n_mpra_val = x_boot_valid.shape[0]
                
            cur_mpra_val_inds = rng.choice(y_boot_valid.shape[0],cur_n_mpra_val,replace=False)
            x_boot_valid = x_boot_valid[cur_mpra_val_inds]
            y_boot_valid = y_boot_valid[cur_mpra_val_inds]

            # train on all the other crossfolds

            x_train = np.vstack([x_cfs[i] for i in range(n_cfs) if i != test_cf_idx and i != val_cf_idx])
            y_train = np.vstack([y_cfs[i] for i in range(n_cfs) if i != test_cf_idx and i != val_cf_idx])
            # randomly downsample to cur_n_mpra
            if cur_n_mpra_train > x_train.shape[0]:
                print(f'cur_n_mpra_train ({cur_n_mpra_train}) > x_train.shape[0] ({x_train.shape[0]}), sampling {x_train.shape[0]}')
                cur_n_mpra_train = x_train.shape[0]

            cur_mpra_inds = rng.choice(y_train.shape[0],cur_n_mpra_train,replace=False)
            x_train = x_train[cur_mpra_inds]
            y_train = y_train[cur_mpra_inds]

            x_dhs_train = np.vstack([dhs_x_cfs[i] for i in range(n_dhs_cfs) if i != dhs_test_cf_idx])
            y_dhs_train = np.vstack([dhs_y_cfs[i] for i in range(n_dhs_cfs) if i != dhs_test_cf_idx])
            # randomly sample cur_n_dhs from the dhs data
            if cur_n_dhs > x_dhs_train.shape[0]:
                print(f'cur_n_dhs ({cur_n_dhs}) > x_dhs_train.shape[0] ({x_dhs_train.shape[0]}), sampling {x_dhs_train.shape[0]}')
                cur_n_dhs = x_dhs_train.shape[0]
            cur_dhs_inds = rng.choice(y_dhs_train.shape[0],cur_n_dhs,replace=False)
            x_dhs_train = x_dhs_train[cur_dhs_inds]
            y_dhs_train = y_dhs_train[cur_dhs_inds]

            # concatenate dhs not in test set to training set
            x_boot_train = np.concatenate([x_train, x_dhs_train], axis=0)
            y_boot_train = np.concatenate([y_train, y_dhs_train], axis=0)

            if boot_idx in skip_inds:
                continue

            ### Add reverse complements to training data ###
            x_boot_train = np.concatenate([
                x_boot_train,
                x_boot_train[:, ::-1, ::-1]
            ], axis=0)

            y_boot_train = np.concatenate([
                y_boot_train,
                y_boot_train
            ], axis=0)

            
            # load the pre-trained model 1-9
            K.clear_session()
            model_name = f'{cf_model_dir}/test_fold_{test_cf_idx}/{model_basename}{val_cf_idx}.h5'
            model = load_model(model_name)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,beta_1=0.9,beta_2=0.999), loss='mse')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, min_delta=min_delta_r,
                                        patience=patience_r, min_lr=lr*1e-3)
            stop_early = EarlyStopping(monitor='val_loss', min_delta = min_delta_se, patience=patience_se,restore_best_weights=True)
            callbacks = [reduce_lr, stop_early]

            history = model.fit(x_boot_train, y_boot_train, batch_size=64, epochs=250, validation_data=(x_boot_valid, y_boot_valid), callbacks=callbacks)

            # Save model and weights

            model_path = f'{save_dir}/boot{boot_idx}'
            os.makedirs(model_path,exist_ok=True)
            model.save(f'{model_path}/{model_basename}{val_cf_idx}_ft_n{cur_n}.h5')
            print('Saved trained model at %s ' % model_path)
            # also save the history
            # import pickle
            with open(f'{model_path}/{results_basename}_n{cur_n}_boot{boot_idx}_history.pkl', 'wb') as f:
                pickle.dump(history.history, f)

    # ...calculate and save performance on the test set...#

            y_test_hat = model.predict(x_test)

            y_preds.append(y_test_hat)

            metrics = ['pearsonr','spearmanr']
            for corr_idx,corr in enumerate([pearsonr,spearmanr]):
                r_h_d2   = corr(y_test[:,0],y_test_hat[:,0])[0]
                r_k_d2   = corr(y_test[:,1],y_test_hat[:,1])[0]
                r_h2k_d2   = corr(y_test[:,0]-y_test[:,1],y_test_hat[:,0]-y_test_hat[:,1])[0]

                
            # print results to text file
            with open(f'{save_dir}/boot{boot_idx}/{results_basename}_{metrics[corr_idx]}.txt', 'a') as f:
                # write the hyperparameters for this model
                f.write(f'{r_h_d2:.3f},{r_k_d2:.3f},{r_h2k_d2:.3f}\n')

        # save the ensemble prediction performance - in a separate save file for ease of reading in later
        if len(y_preds) == 0:
            continue
        
        y_preds = np.stack(y_preds)
        y_preds_mean = np.mean(y_preds,axis=0)

        for corr_idx,corr in enumerate([pearsonr,spearmanr]):
            r_h_d2   = corr(y_test[:,0],y_preds_mean[:,0])[0]
            r_k_d2   = corr(y_test[:,1],y_preds_mean[:,1])[0]
            r_h2k_d2   = corr(y_test[:,0]-y_test[:,1],y_preds_mean[:,0]-y_preds_mean[:,1])[0]

            with open(f'{save_dir}/{results_basename}_ensemble_{metrics[corr_idx]}.txt', 'a') as f:
                f.write(f'{r_h_d2:.3f},{r_k_d2:.3f},{r_h2k_d2:.3f}\n')