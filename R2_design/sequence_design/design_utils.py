import pandas as pd
import numpy as np
import editdistance

TEST_FOLD_IDX = 0 # keep fixed
DESIGNED_SEQ_LEN = 145 # keep fixed
MODEL_TYPE_DICT = {
    'd1_finetuned': ('d1_wide_ft_cf_t0_v',200,6.423,-4.027,4.756,-2.648,5.732,-2.401,9),
    'd2_dhs': ('d2_dhs_wide_cf_t0_v',200,6.210,-4.552,4.693,-2.653,5.191,-2.392,9),
}
MODEL_TYPE_DF = pd.DataFrame.from_dict(MODEL_TYPE_DICT,orient='index',columns=['model_basename','input_len','max_pred_h2k','max_pred_k2h','max_pred_hepg2','min_pred_hepg2','max_pred_k562','min_pred_k562','n_ensemble'])
CELL_TYPES = ['HEPG2','K562']


import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Average
import collections.abc as collections

def load_ensemble_model(model_dir,model_basename,model_inds,seq_length,max_seq_len=200,model_basename_suffix=''):

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
        models[i] = load_model(f'{model_dir}/{model_basename}{model_inds[i]}{model_basename_suffix}.h5')
        models[i]._name = f"model_idx{i}"

    ensemble_input = Input(shape=(seq_length,4))
    ensemble_outputs = [model(input_padding_layer(ensemble_input)) for model in models]
    ensemble_avg = Average()(ensemble_outputs)
    ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_avg)

    return ensemble_model


def load_maxmin_ensemble_model(model_dir,model_basename,model_inds,seq_length,tgt_cell_type,max_seq_len=200,model_basename_suffix=''):
    
    # # Layer for padding input to resnet model
    # input_padding_layer = layers.Lambda(
    #     lambda x: tensorflow.pad(
    #         x,
    #         [[0, 0], [max_seq_len - seq_length,0], [0, 0]],
    #         "CONSTANT",
    #         constant_values=0,
    #     ),
    #     name='input_padding_layer',
    # )
    # do centering instead of left padding
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

    # Outputs of individual models
    model_input = layers.Input(shape=(seq_length, 4), name='model_input')
    individual_models = [None] * len(model_inds)
    for i in range(len(model_inds)):
        individual_models[i] = load_model(f'{model_dir}/{model_basename}{model_inds[i]}{model_basename_suffix}.h5')
        individual_models[i]._name = f"model_idx{i}"
    models_individual_output = [m(input_padding_layer(model_input))[0] for m in individual_models]
    
    # Layer for selecting output
    output_idx_to_maximize = CELL_TYPES.index(tgt_cell_type)
    mask_min = np.zeros((1, len(CELL_TYPES)))
    mask_min[:, output_idx_to_maximize] = 1
    mask_min = tensorflow.cast(mask_min, tensorflow.float32)
    mask_max = np.ones((1, len(CELL_TYPES)))
    mask_max[:, output_idx_to_maximize] = 0
    mask_max = tensorflow.cast(mask_max, tensorflow.float32)
    select_output_layer = layers.Lambda(
        lambda x: tensorflow.reduce_min(tensorflow.stack(x, axis=-1), axis=-1)*mask_min + \
            tensorflow.reduce_max(tensorflow.stack(x, axis=-1), axis=-1)*mask_max,
        name='select_output_layer',
    )
    model_ensemble_output = select_output_layer(models_individual_output)

    model_ensemble = Model(
        model_input,
        model_ensemble_output,
    )

    return model_ensemble




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

# calculate the longest contiguous subsequence of the same nucleotide in a sequence
def longest_repeat(seq):
    # seq is a string
    # return the length of the longest contiguous repeat
    # if no repeats, return 0
    # if seq is empty, return 0
    if len(seq) == 0:
        return 0
    else:
        # initialize
        max_repeat = 1
        current_repeat = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_repeat += 1
            else:
                current_repeat = 1
            if current_repeat > max_repeat:
                max_repeat = current_repeat
        return max_repeat
    
def get_paired_editdistances(seqs):
    shuffle_index = np.arange(len(seqs))
    
    # Reject shufflings if any element remains in its original position
    while np.any(shuffle_index==np.arange(len(seqs))):
        np.random.shuffle(shuffle_index)

    distances = []
    for i in range(len(seqs)) :
        if i == shuffle_index[i] :
            continue
        seq_1 = seqs[i]
        seq_2 = seqs[shuffle_index[i]]
        dist = editdistance.eval(seq_1, seq_2)
        distances.append(dist)

    distances = np.array(distances) / len(seqs[0])

    return distances