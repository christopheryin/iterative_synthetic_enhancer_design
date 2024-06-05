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


model_basename = 'd1_wide_cf_t0_v'
results_basename = 'd1_wide_cf10_finetuned' ######################## change this!!!!!!!!!!!

### Define model and training hyperparameters ###
n_epochs = 100
batch_size = 64

# load all the crossfolds for d2 and dhs
cf_dir = 'cf10'

n_cfs = 10
x_cfs = [np.load(f'{cf_dir}/x_cf{i}.npy') for i in range(n_cfs)]
y_cfs = [np.load(f'{cf_dir}/y_cf{i}.npy') for i in range(n_cfs)]
w_cfs = [np.load(f'{cf_dir}/w_cf{i}.npy') for i in range(n_cfs)]

# load dhs cfs
n_dhs_cfs = 5
dhs_x_cfs = [np.load(f'{cf_dir}/dhs/dhs_x_cf{i}.npy') for i in range(n_dhs_cfs)]
dhs_y_cfs = [np.load(f'{cf_dir}/dhs/dhs_y_cf{i}.npy') for i in range(n_dhs_cfs)]
dhs_test_cf_idx = 0

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


# for test_cf in range(n_crossfolds):
# only use 1 crossfold for now so I don't train 90 models unnecessarily
# that said...ugh how to account for ensembles here? I guess I'll just have one ensemble at the end...
for test_cf_idx in [0]:

    x_test = x_cfs[test_cf_idx]
    y_test = y_cfs[test_cf_idx]

    # # concatenate dhs to the test set
    # x_test = np.concatenate([x_test, dhs_x_cfs[dhs_test_cf_idx]], axis=0)
    # y_test = np.concatenate([y_test, dhs_y_cfs[dhs_test_cf_idx]], axis=0)
    x_test_dhs = dhs_x_cfs[dhs_test_cf_idx]
    y_test_dhs = dhs_y_cfs[dhs_test_cf_idx]

    y_preds = []
    y_preds_dhs = []

    for val_cf_idx in [i for i in range(n_cfs) if i != test_cf_idx]:

        results_basename = f'cf10_d1_t{test_cf_idx}_finetuned_training_results'

        x_valid = x_cfs[val_cf_idx]
        y_valid = y_cfs[val_cf_idx]

        # train on all the other crossfolds

        x_train = np.vstack([x_cfs[i] for i in range(n_cfs) if i != test_cf_idx and i != val_cf_idx])
        y_train = np.vstack([y_cfs[i] for i in range(n_cfs) if i != test_cf_idx and i != val_cf_idx])

        # concatenate dhs not in test set to training set
        x_train = np.concatenate([x_train, np.vstack([dhs_x_cfs[i] for i in range(n_dhs_cfs) if i != dhs_test_cf_idx])], axis=0)
        y_train = np.concatenate([y_train, np.vstack([dhs_y_cfs[i] for i in range(n_dhs_cfs) if i != dhs_test_cf_idx])], axis=0)

        # add reverse complements to the training set
        x_train = np.concatenate([
            x_train,
            x_train[:, ::-1, ::-1]
        ], axis=0)

        y_train = np.concatenate([
            y_train,
            y_train
        ], axis=0)

        # frees up memory
        K.clear_session()
        model_dir = 'cf_model_dir/test_fold_0/d1'
        model_basename = f'd1_wide_cf_t{test_cf_idx}_v{val_cf_idx}'
        model_name = f'{model_dir}/{model_basename}.h5'
        model = load_model(model_name)

        # redefine with best values from hp search ##############################################################################################
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

        # Save model and weights
        save_dir = f'cf_model_dir/test_fold_{test_cf_idx}/d1_finetuned'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, model_basename + '_ft.h5')
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        y_test_hat = model.predict(x_test)
        y_test_hat_dhs = model.predict(x_test_dhs)
        y_preds += [y_test_hat]
        y_preds_dhs += [y_test_hat_dhs]

        metrics = ['pearsonr','spearmanr']
        for corr_idx,corr in enumerate([pearsonr,spearmanr]):
            r_h_d2   = corr(y_test[:,0],y_test_hat[:,0])[0]
            r_k_d2   = corr(y_test[:,1],y_test_hat[:,1])[0]
            r_h2k_d2   = corr(y_test[:,0]-y_test[:,1],y_test_hat[:,0]-y_test_hat[:,1])[0]

            r_h_dhs = corr(y_test_dhs[:,0],y_test_hat_dhs[:,0])[0]
            r_k_dhs = corr(y_test_dhs[:,1],y_test_hat_dhs[:,1])[0]
            r_h2k_dhs = corr(y_test_dhs[:,0]-y_test_dhs[:,1],y_test_hat_dhs[:,0]-y_test_hat_dhs[:,1])[0]
                
            # print results to text file
            with open(f'{save_dir}/{results_basename}_{metrics[corr_idx]}.txt', 'a') as f:
                # write the hyperparameters for this model
                f.write(f'{r_h_d2:.3f},{r_k_d2:.3f},{r_h2k_d2:.3f},{r_h_dhs},{r_k_dhs},{r_h2k_dhs}\n')

    # average all the y_preds for this test fold
    y_preds = np.array(y_preds)
    y_test_hat = np.mean(y_preds, axis=0)
    y_preds_dhs = np.array(y_preds_dhs)
    y_test_hat_dhs = np.mean(y_preds_dhs, axis=0)

    metrics = ['pearsonr','spearmanr']
    for corr_idx,corr in enumerate([pearsonr,spearmanr]):
        r_h_d2   = corr(y_test[:,0],y_test_hat[:,0])[0]
        r_k_d2   = corr(y_test[:,1],y_test_hat[:,1])[0]
        r_h2k_d2   = corr(y_test[:,0]-y_test[:,1],y_test_hat[:,0]-y_test_hat[:,1])[0]

        r_h_dhs = corr(y_test_dhs[:,0],y_test_hat_dhs[:,0])[0]
        r_k_dhs = corr(y_test_dhs[:,1],y_test_hat_dhs[:,1])[0]
        r_h2k_dhs = corr(y_test_dhs[:,0]-y_test_dhs[:,1],y_test_hat_dhs[:,0]-y_test_hat_dhs[:,1])[0]
            
        # print results to text file
        with open(f'{save_dir}/{results_basename}_{metrics[corr_idx]}.txt', 'a') as f:
            # write the hyperparameters for this model
            f.write('\n')
            f.write(f'{r_h_d2:.3f},{r_k_d2:.3f},{r_h2k_d2:.3f},{r_h_dhs},{r_k_dhs},{r_h2k_dhs}\n')
            f.write('\n')