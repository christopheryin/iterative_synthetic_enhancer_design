import json
import os
import pickle
import random
import sys

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import seaborn

from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Average
import tensorflow.keras.backend as K

import collections.abc as collections # apparently vanilla collections has a bug now and was triggering the assert in load_ensemble_model erroneously

import Bio
import Bio.Seq
import Bio.SeqRecord
import Bio.SeqIO
import Bio.SeqUtils

import corefsp


import argparse
import seaborn as sns


import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model, load_model
# import Average
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, GlobalMaxPooling1D, concatenate, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute
from tensorflow.keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply, Average
import tensorflow.keras.backend as K


def load_ensemble_model(model_dir,model_basename,model_inds,seq_length,max_seq_len=200,model_suffix=''):

    assert isinstance(model_inds, collections.Iterable), "model_inds must be a list or array"

    # Layer for padding input to resnet model
    # input_padding_layer = layers.Lambda(
    #     lambda x: tensorflow.pad(
    #         x,
    #         [[0, 0], [max_seq_len - seq_length,0], [0, 0]], # CHANGING THIS TO PAD THE 5' END INSTEAD OF 3' !!!!!!!!!!!!!!!! THIS MAKES MORE SENSE
    #         "CONSTANT",
    #         constant_values=0,
    #     ),
    #     name='input_padding_layer',
    # )
    lpad = (max_seq_len - seq_length)//2
    rpad = max_seq_len - seq_length - lpad
    input_padding_layer = layers.Lambda(
        lambda x: tensorflow.pad(
            x,
            [[0, 0], [lpad,rpad], [0, 0]],
            "CONSTANT",
            constant_values=0,
        ),
        name='input_padding_layer',
    )

    models = [None] * len(model_inds)
    for i in range(len(model_inds)):
        models[i] = load_model(f'{model_dir}/{model_basename}{model_inds[i]}{model_suffix}.h5')
        models[i]._name = f"model_idx{i}"

    ensemble_input = Input(shape=(seq_length,4))
    ensemble_outputs = [model(input_padding_layer(ensemble_input)) for model in models]
    ensemble_avg = Average()(ensemble_outputs)
    ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_avg)

    return ensemble_model

############################################################################################################

matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

### Design Run Parameters ###
DESIGN_TYPE = 'fsp' # hardcoded for this script
PALETTE = {'HEPG2': 'tab:orange', 'K562': 'tab:blue'}
n_seqs_per_target = 200 # design 1000 sequences
n_seqs_to_save = 100 # save the top 100
seq_length = 145

# Round 1: target weight is one, design for all
target_weight = 1.0
cell_types_to_design = ['HEPG2', 'K562']
CELL_TYPES = ['HEPG2', 'K562']

def generate_sequences(model_basename,design_dir,model_dir,design_seq_len=145,target_function='h2k',suffix='',target_weight=1.0,seed=0,):

    # Load models
    max_seq_len = 200 # hardcode this for now
    n_models_to_ensemble = 10 # hardcode this for now

    for cell_type_idx, cell_type in enumerate(CELL_TYPES):

        output_filepath = f'{design_dir}/designed{suffix}_{DESIGN_TYPE}_{cell_type}_seqs.fasta'
        if os.path.exists(output_filepath):
            print(f"Generating sequences for cell type {cell_type} already exist. Skipping.")
            continue
        
        if cell_type not in cell_types_to_design:
            continue
        print(f"Generating sequences for cell type {cell_type} ({cell_type_idx + 1} / {len(CELL_TYPES)})...")

        # Make model

        model_ensemble = load_ensemble_model(model_dir,model_basename,range(1,n_models_to_ensemble),
                                            design_seq_len,max_seq_len=max_seq_len,model_suffix=suffix)

        # this output maximizes cell type specificity
        def target_loss_func_h2k(model_preds):
            # model_preds has dimensions (n_seqs, n_outputs)
            target_score = - tensorflow.reduce_mean(model_preds[:, cell_type_idx]-model_preds[:, 1-cell_type_idx])
            return target_score
            
        # Define PWM loss
        # The following is supposed to penalize repeats
        def pwm_loss_func(pwm):
            # PWM has dimensions (n_seqs, seq_length, n_channels)
            # return tensorflow.reduce_mean(pwm[:, :-2, :] * pwm[:, 1:-1, :] * pwm[:, 2:, :])
            return tensorflow.reduce_mean(pwm[:, :-1, :] * pwm[:, 1:, :])
        
        target_loss_func = target_loss_func_h2k

        seq_vals, pred_vals, train_hist = corefsp.design_seqs(
            model_ensemble,
            target_loss_func,
            pwm_loss_func=pwm_loss_func,
            seq_length=design_seq_len,
            n_seqs=n_seqs_per_target,
            target_weight=target_weight,
            pwm_weight=2.5, # was previously 3
            entropy_weight=1e-3,
            learning_rate=0.001,
            n_iter_max=1000,
            init_seed=seed,
        )

        # sort generated preds descending by generated_preds[:,cell_type_idx] - generated_preds[:,1-cell_type_idx]
        sort_inds = np.argsort(pred_vals[:,cell_type_idx] - pred_vals[:,1-cell_type_idx])[::-1][:n_seqs_to_save] # this escerpts the top n_seqs_to_save that were designed
        seq_vals = seq_vals[sort_inds]
        pred_vals = pred_vals[sort_inds]


        # Save training history data
        train_hist_filepath = os.path.join(design_dir, f"designed{suffix}_{DESIGN_TYPE}_cell_type_{cell_type}_train_hist.pickle")
        with open(train_hist_filepath, 'wb') as handle:
            pickle.dump(train_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save sequences
        seqs = []
        alphabet = ['A', 'C', 'G', 'T']
        seqs_index = np.argmax(seq_vals, axis=-1)
        for seq_index in seqs_index:
            seqs.append(''.join([alphabet[idx] for idx in seq_index]))
        seq_records = []
        for seq_idx, seq in enumerate(seqs):
            seq_id = f'designed{suffix}_{DESIGN_TYPE}_target_{cell_type}_seq_{seq_idx}'
            seq_records.append(Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq), id=seq_id, description=""))
        with open(output_filepath, "w") as output_handle:
            Bio.SeqIO.write(seq_records, output_handle, "fasta")

        print(f"Done with cell type {cell_type}.")

    print("Done.")


# I guess I should also include the n = 0, i.e. the M0 models, as debugging comparison; or could drop the n even further...
# I think I want to revert the train/val bootstrap separation which seemed to not work well, can see it having too much repetition within each split
nvars = [100,200,300,500,1000,1350,1750][:-1]
# nvars = ['M1']

base_model_dir = 'nvar_model_dir_cf10' # v4 = splitting dhs and mpra data at constant ratio

design_seq_len = 145
# generate sequences for both cell types!

for boot_idx in [0,1,2,3,4]:
    seed=99 # was previously 0
    base_design_dir = f'{base_model_dir}/designs_boot{boot_idx}_seed{seed}'

    for cur_n in nvars:

        model_dir = f'{base_model_dir}/n{cur_n}/boot{boot_idx}/cf_models'
        model_basename = f'd1_wide_cf_t0_v'
        model_suffix= f'.h5_ft_n{cur_n}'

        if not os.path.exists(model_dir):
            print(f"Model directory {model_dir} does not exist. Skipping.")
            continue
        
        for target_weight in [1]:

            design_dir = f'{base_design_dir}/target{target_weight}'
            fname = f'{design_dir}/designed.h5_ft_n{cur_n}_fsp_HEPG2_seqs.fasta'
            if os.path.exists(fname):
                continue
            # design_dir = f'{base_design_dir}'
            os.makedirs(design_dir, exist_ok=True)
            generate_sequences(model_basename,design_dir,model_dir,design_seq_len=145,
                            target_function='h2k',suffix=f'{model_suffix}',
                            target_weight=target_weight,seed=seed)
