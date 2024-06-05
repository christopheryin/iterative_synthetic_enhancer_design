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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import tensorflow.keras.losses

def build_wide_model(n_filters, filt_sizes, n_dense, dropout_rate, n_classes,l2_weight,seq_len=145):
    sequence_input = Input(shape=(seq_len, 4),name="pat_input")

    convs = [None]*len(filt_sizes)

    for i in range(len(filt_sizes)):
        conv1           = Conv1D(n_filters, filt_sizes[i], padding='same', activation='linear', name = "pat_conv_" + str(i))(sequence_input)
        batchnorm1      = BatchNormalization(axis=-1,name = "pat_batchnorm_" + str(i))(conv1)
        exp1            = Activation('exponential',name = "pat_relu_" + str(i))(batchnorm1)
        convs[i]        = Dropout(dropout_rate,name = "pat_dropout_" + str(i))(GlobalMaxPooling1D(name = "pat_pool_" + str(i))(exp1))

    concat1           = concatenate(convs,name="pat_concat_layer")

    l2_reg = tf.keras.regularizers.l2(l2_weight)
    dense           = Dense(n_dense,activation='relu',name="pat_dense",kernel_regularizer=l2_reg)(concat1)

    output          = Dense(n_classes,activation='linear',name="pat_output")(dense)

    model = Model(inputs=sequence_input, outputs=output)

    return model

model_basename = 'd1_wide'
results_basename = 'd1_wide' ######################## change this!!!!!!!!!!!

### Define model and training hyperparameters ###
n_epochs = 100
batch_size = 64

n_filters = 608
filt_sizes = [11,15,21]
n_dense = 224
dropout_rate = 0.181
n_classes = 2
n_datasets = 2
lr = 3e-4
l2_weight = 1e-3

# optimized on D1 only
hp1 = {
    'n_filters': 608,
    'filt_sizes': [11,15,21],
    'n_dense': 224,
    'dropout_rate': 0.181,
    'l2_weight': 1e-3,
    'lr': 3e-4
}

n_classes = 2

# load all the crossfolds - each split stored in separate .npy file
cf_dir = 'cf10'

n_cfs = 10
x_cfs = [np.load(f'{cf_dir}/x_cf{i}.npy') for i in range(n_cfs)]
y_cfs = [np.load(f'{cf_dir}/y_cf{i}.npy') for i in range(n_cfs)]
# binary matrix indicating data source where col0 = 1 indicates R1-MPRA, col1 = 1 indicates R1-DHS
w_cfs = [np.load(f'{cf_dir}/w_cf{i}.npy') for i in range(n_cfs)] 

# padd to 200 bp, also extract only the d1 columns
lpad = (200-145)//2
rpad = 200 - 145 - lpad

# for each cf, extract the y_cf rows where w_cf[:,0] == 1 (these are the d1 seqs)
for cf_idx in range(n_cfs):
    x_cfs[cf_idx] = x_cfs[cf_idx][w_cfs[cf_idx][:,0] == 1]
    # pad the x_cfs with all 0s by lpad on the left, rpad on the right (axis 1)
    x_cfs[cf_idx] = np.pad(x_cfs[cf_idx], ((0,0),(lpad,rpad),(0,0)), 'constant', constant_values=0)

    y_cfs[cf_idx] = y_cfs[cf_idx][w_cfs[cf_idx][:,0] == 1]
    # extract the 0st and 2nd columns of y_cf (d2)
    y_cfs[cf_idx] = y_cfs[cf_idx][:,[0,2]]

for hp_idx,hp in enumerate([hp1]):
    # only use 1 crossfold for now so I don't train 90 models unnecessarily
    for test_cf_idx in [0]:

        x_test = x_cfs[test_cf_idx]
        y_test = y_cfs[test_cf_idx]

        y_preds = []

        for val_cf_idx in [i for i in range(n_cfs) if i != test_cf_idx]:

            x_valid = x_cfs[val_cf_idx]
            y_valid = y_cfs[val_cf_idx]

            # train on all the other crossfolds

            x_train = np.vstack([x_cfs[i] for i in range(n_cfs) if i != test_cf_idx and i != val_cf_idx])
            y_train = np.vstack([y_cfs[i] for i in range(n_cfs) if i != test_cf_idx and i != val_cf_idx])

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
            model_name = f'{model_basename}_cf_t{test_cf_idx}_v{val_cf_idx}'

            model = build_wide_model(hp['n_filters'], hp['filt_sizes'], hp['n_dense'], hp['dropout_rate'], n_classes, hp['l2_weight'],seq_len=200)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp['lr'],beta_1=0.9,beta_2=0.999), loss='mse')

            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=1e-6,
                    patience=5,
                    verbose=True,
                    restore_best_weights=True
                )
            ]

            train_history = model.fit(
                [x_train],
                [y_train],
                shuffle=True,
                epochs=n_epochs,
                batch_size=batch_size,
                validation_data=(
                    [x_valid],
                    [y_valid]
                ),
                callbacks=callbacks
            )

            # Save model and weights
            save_dir = f'cf_model_dir/test_fold_{test_cf_idx}/hp_{hp_idx}'

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            model_path = os.path.join(save_dir, model_name + '.h5')
            model.save(model_path)
            print('Saved trained model at %s ' % model_path)

            y_test_hat = model.predict(x_test)
            y_preds += [y_test_hat]
            r_h,_   = pearsonr(y_test[:,0],y_test_hat[:,0])
            r_k,_   = pearsonr(y_test[:,1],y_test_hat[:,1])
            r_h2k,_   = pearsonr(y_test[:,0]-y_test[:,1],y_test_hat[:,0]-y_test_hat[:,1])
            
            # print results to text file
            with open(f'{save_dir}/cf10_d1_t{test_cf_idx}_training_results.txt', 'a') as f:
                # write the hyperparameters for this model
                # f.write(f'blks_per_group: {n_blocks_per_group_vals[i]} | filt_size: {filt_size_vals[i]} | dilations: {dilation_vals[i]} | lr: {lr} | n_filters: {n_filters}\n')
                f.write(f'{r_h:.3f},{r_k:.3f},{r_h2k:.3f}\n')

        # average all the y_preds for this test fold
        y_preds = np.array(y_preds)
        y_test_hat = np.mean(y_preds, axis=0)

        r_h,_   = pearsonr(y_test[:,0],y_test_hat[:,0])
        r_k,_   = pearsonr(y_test[:,1],y_test_hat[:,1])
        r_h2k,_   = pearsonr(y_test[:,0]-y_test[:,1],y_test_hat[:,0]-y_test_hat[:,1])

        with open(f'{save_dir}/cf10_d1_t{test_cf_idx}_training_results.txt', 'a') as f:
            # write the hyperparameters for this model
            f.write('\n')
            f.write(f'{r_h:.4f},{r_k:.4f},{r_h2k:.4f}\n')
            f.write('\n')