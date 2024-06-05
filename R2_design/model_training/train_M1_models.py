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

def build_wide_model(n_filters, filt_sizes, n_dense, dropout_rate, n_classes,l2_weight):
    sequence_input = Input(shape=(145, 4),name="pat_input")

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


def test_model(model, model_idx,cf_idx,x_test_remerged, y_test_remerged, n_d1,fname,model_dir):
    """_summary_

    Args:
        model (torch model): Should be the widemodel buried within the Multidata object (i.e. have output dim = 2)
        test_loader (_type_): _description_
        y_test_remerged (_type_): _description_
        n_d1 (_type_): _description_
        device (_type_): _description_
        fname (_type_): _description_
    """


    y_test_hat = model.predict(x_test_remerged)

    y_test_hat_hepg2_d1 = y_test_hat[:n_d1,0]
    y_test_hat_hepg2_d2 = y_test_hat[n_d1:,0]
    y_test_hat_k562_d1  = y_test_hat[:n_d1,1]
    y_test_hat_k562_d2  = y_test_hat[n_d1:,1]
    y_test_hat_h2k_d1 = y_test_hat_hepg2_d1 - y_test_hat_k562_d1
    y_test_hat_h2k_d2 = y_test_hat_hepg2_d2 - y_test_hat_k562_d2

    r_hepg2_d1,_ = spearmanr(y_test_remerged[:n_d1,0],  y_test_hat_hepg2_d1)
    r_hepg2_d2,_ = spearmanr(y_test_remerged[n_d1:,0],  y_test_hat_hepg2_d2)
    r_k562_d1,_  = spearmanr(y_test_remerged[:n_d1,1],  y_test_hat_k562_d1)
    r_k562_d2,_  = spearmanr(y_test_remerged[n_d1:,1],  y_test_hat_k562_d2)
    r_h2k_d1,_   = spearmanr(y_test_remerged[:n_d1,0] - y_test_remerged[:n_d1,1], y_test_hat_h2k_d1)
    r_h2k_d2,_   = spearmanr(y_test_remerged[n_d1:,0] - y_test_remerged[n_d1:,1], y_test_hat_h2k_d2)

    
    # combine hepg2 results from both datasets
    y_test_hat_hepg2 = np.concatenate([y_test_hat_hepg2_d1, y_test_hat_hepg2_d2])
    r_hepg2,_ = spearmanr(y_test_remerged[:,0], y_test_hat_hepg2)

    # combine k562 results from both datasets
    y_test_hat_k562 = np.concatenate([y_test_hat_k562_d1, y_test_hat_k562_d2])
    r_k562,_ = spearmanr(y_test_remerged[:,1], y_test_hat_k562)

    # combine h2k results from both datasets
    y_test_hat_h2k = np.concatenate([y_test_hat_h2k_d1, y_test_hat_h2k_d2])
    r_h2k,_ = spearmanr(y_test_remerged[:,0]-y_test_remerged[:,1], y_test_hat_h2k)

    print(f'HEPG2 d1: {r_hepg2_d1}')
    print(f'HEPG2 d2: {r_hepg2_d2}')
    print(f'K562 d1: {r_k562_d1}')
    print(f'K562 d2: {r_k562_d2}')
    print(f'HEPG2-K562 d1: {r_h2k_d1}')
    print(f'HEPG2-K562 d2: {r_h2k_d2}')
    # print results for combined datasets
    print(f'HEPG2: {r_hepg2}')
    print(f'K562: {r_k562}')
    print(f'HEPG2-K562: {r_h2k}')

    # check if any of the r values are negative
    if r_hepg2_d1 < 0 or r_hepg2_d2 < 0 or r_k562_d1 < 0 or r_k562_d2 < 0 or r_h2k_d1 < 0 or r_h2k_d2 < 0:
        print('Negative r value detected')
        return -1
    else:
        with open(fname, 'a') as f:
            f.write(f'{model_idx},{r_hepg2_d1:8.3f},{r_hepg2_d2:8.3f},{r_k562_d1:7.3f},{r_k562_d2:7.3f},{r_h2k_d1:13.3f},{r_h2k_d2:13.3f},{r_hepg2:5.3f},{r_k562:6.3f},{r_h2k:9.3f}\n')
            # also save y_test_hat
            np.save(f'{model_dir}/y_test_hat_{model_basename}_{cf_idx}_{model_idx}.npy', y_test_hat)
    return 0

# Save locations
model_dir = 'retrained_crossfold_models'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

model_basename = 'final_retrained'
results_basename = 'final_retrained' ######################## change this!!!!!!!!!!!

### Define model and training hyperparameters ###
n_epochs = 100
batch_size = 64

n_filters = 608
filt_sizes = [11,15,21]
n_dense = 224
dropout_rate = 0.181
n_classes = 2
n_datasets = 2
lr = 0.0003
l2_weight = 0.001

# load all the crossfolds
cf_dir = 'crossfolds'

n_crossfolds = 5
x_cfs = [np.load(f'{cf_dir}/x_cf{i}.npy') for i in range(n_crossfolds)]
y_cfs = [np.load(f'{cf_dir}/y_cf{i}.npy') for i in range(n_crossfolds)]
w_cfs = [np.load(f'{cf_dir}/w_cf{i}.npy') for i in range(n_crossfolds)]

n_models = 10 # number of models to train per crossfold

for test_cf in range(n_crossfolds):

    # create folder f'{model_dir}/cf{test_cf}' if it doesn't exist
    cf_model_dir = f'{model_dir}/cf{test_cf}'
    if not os.path.isdir(cf_model_dir):
        os.makedirs(cf_model_dir)

    results_fname = f'{cf_model_dir}/{results_basename}_cf{test_cf}.csv'

    x_train = np.vstack([x_cfs[i] for i in range(n_crossfolds) if i != test_cf])
    y_train = np.vstack([y_cfs[i] for i in range(n_crossfolds) if i != test_cf])
    w_train = np.vstack([w_cfs[i] for i in range(n_crossfolds) if i != test_cf])

    # add reverse complements to the training set
    x_train = np.concatenate([
        x_train,
        x_train[:, ::-1, ::-1]
    ], axis=0)

    y_train = np.concatenate([
        y_train,
        y_train
    ], axis=0)

    w_train = np.concatenate([
        w_train,
        w_train
    ], axis=0)

    x_test = x_cfs[test_cf]
    y_test = y_cfs[test_cf]
    w_test = w_cfs[test_cf]

    # assemble test set for model, rather than model_to_fit
    ctrl_inds = np.sum(w_test[:,:2], axis=1)==2

    x_merged_test = x_test[~ctrl_inds]
    y_merged_test = y_test[~ctrl_inds]
    w_merged_test = w_test[~ctrl_inds]

    data1_inds = w_merged_test[:,0]==1
    data2_inds = w_merged_test[:,1]==1

    y_test_d1 = np.vstack((y_merged_test[data1_inds,0],y_merged_test[data1_inds,2])).T
    y_test_d2 = np.vstack((y_merged_test[data2_inds,1],y_merged_test[data2_inds,3])).T

    x_test_d1 = x_merged_test[data1_inds]
    x_test_d2 = x_merged_test[data2_inds]

    x_test_remerged = np.concatenate((x_test_d1,x_test_d2), axis=0)
    y_test_remerged = np.concatenate((y_test_d1,y_test_d2), axis=0)

    n_d1 = x_test_d1.shape[0]
    n_d2 = x_test_d2.shape[0]


    # write print("HEPG2 D1 | HEPG2 D2 | K562 D1 | K562 D2 | HEPG2-K562 D1 | HEPG2-K562 D2 | HEPG2 |  K562  | HEPG2-K562") to first line of results.txt
    with open(results_fname, 'w') as f:
        f.write("Model,HEPG2 D1,HEPG2 D2,K562 D1,K562 D2,HEPG2-K562 D1,HEPG2-K562 D2,HEPG2,K562,HEPG2-K562\n")

    for model_idx in range(n_models):

        print(f'Beginning model {model_idx}...')

        model_check = -1

        while model_check == -1:

            K.clear_session()

            model = build_wide_model(n_filters, filt_sizes, n_dense, dropout_rate, n_classes,l2_weight)

            # New! add a linear node for the controls to interpolate between df1 and df2 nevermind!!!!!!!!!!!!!!!!
            cell_line_outputs = []
            for cell_line_idx in range(n_classes):

                cell_line_output = Lambda(
                    lambda x: tf.expand_dims(x,axis=-1),
                    name=f'pat_output_{cell_line_idx}'
                )(model.output[:,cell_line_idx])

                cell_line_scale_layer = Dense(
                    1,
                    activation='linear',
                    use_bias=True,
                    name=f'dense_output_scale_{cell_line_idx}'
                )

                cell_line_output_scaled = cell_line_scale_layer(cell_line_output)

                # Concatenate: [original_output] + scaled_outputs
                cell_line_output_full = Concatenate(
                    name=f'concatenate_output_{cell_line_idx}'
                )([cell_line_output, cell_line_output_scaled])

                cell_line_outputs.append(cell_line_output_full)

            model_to_fit_output = Concatenate()(cell_line_outputs)
        
            # Custom loss function
            def weighted_mse_loss(y_true, y_pred):
                sq_diff = tf.square(y_true - y_pred)
                sq_weighted_diff = sq_diff*model_weights_input
                # return tf.reduce_sum(sq_weighted_diff)
                return tf.reduce_mean(sq_weighted_diff)
            
            model_labels_input = Input(shape=(n_classes*n_datasets,))
            model_weights_input = Input(shape=(n_classes*n_datasets,))
            
            custom_loss_to_add = weighted_mse_loss(model_to_fit_output, model_labels_input)

            model_to_fit = Model(
                inputs=[model.input, model_labels_input, model_weights_input],
                outputs=model_to_fit_output
            )

            # Add custom loss function and compile
            model_to_fit.add_loss(custom_loss_to_add)
            model_to_fit.compile(loss=None, optimizer=tf.keras.optimizers.Adam(lr))

            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=1e-6,
                    patience=5,
                    verbose=True,
                    restore_best_weights=True
                )
            ]

            train_history = model_to_fit.fit(
                [x_train,y_train,w_train],
                y_train,
                shuffle=True,
                epochs=n_epochs,
                batch_size=batch_size,
                validation_data=([x_test, y_test, w_test], y_test),
                callbacks=callbacks
            )

            model_check = test_model(model,model_idx,test_cf,x_test_remerged, y_test_remerged, n_d1,results_fname,cf_model_dir)

        # save model
        model.save(f'{cf_model_dir}/{model_basename}_cf{test_cf}_m{model_idx}.h5')