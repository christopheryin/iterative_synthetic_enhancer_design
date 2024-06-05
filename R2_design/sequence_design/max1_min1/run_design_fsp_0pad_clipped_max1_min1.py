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

import tensorflow.keras
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

# sys.path.append('../../../../utils/')
# import seq_utils
sys.path.append('../')
import design_plots
from design_utils import TEST_FOLD_IDX,DESIGNED_SEQ_LEN,MODEL_TYPE_DF,CELL_TYPES
from design_utils import load_ensemble_model, load_maxmin_ensemble_model, seq_to_one_hot, longest_repeat, get_paired_editdistances

import argparse
import seaborn as sns

############################################################################################################

matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

### Design Run Parameters ###
DESIGN_TYPE = 'fsp_clipped' # hardcoded for this script
PALETTE = {'HEPG2': 'tab:orange', 'K562': 'tab:blue'}
n_seqs_per_target = 100
seq_length = 145

# Round 1: target weight is one, design for all
target_weight = 1.0
cell_types_to_design = CELL_TYPES

def generate_sequences_target(model_type,design_seq_len=145,target_function='h2k',suffix='max1',use_maxmin=False):

    # Load models
    model_basename = MODEL_TYPE_DF.loc[model_type]['model_basename']
    max_seq_len = MODEL_TYPE_DF.loc[model_type]['input_len']
    n_models_to_ensemble = MODEL_TYPE_DF.loc[model_type]['n_ensemble']

    # generate target vector based on max and min predictions: target 5 values uniformly along range of 0 to max for each cell type
    max_pred_hepg2 = MODEL_TYPE_DF.loc[model_type]['max_pred_hepg2']
    max_pred_k562 = MODEL_TYPE_DF.loc[model_type]['max_pred_k562'] # this is already negative, keep sign as is
    min_pred_hepg2 = MODEL_TYPE_DF.loc[model_type]['min_pred_hepg2']
    min_pred_k562 = MODEL_TYPE_DF.loc[model_type]['min_pred_k562'] # this is already negative, keep sign as is

    if suffix == 'max1':
        hepg2_target = max_pred_hepg2*1.1
        k562_target = max_pred_k562*1.1
    elif suffix == 'min1':
        hepg2_target = min_pred_hepg2*1.1
        k562_target = min_pred_k562*1.1
    else:
        print("ERROR: suffix must be 'max1' or 'min1'")
        return

    targets = [hepg2_target,k562_target]

    # maximize 1 cell type, not cell type difference

    # Directory depends on target weight
    if design_seq_len < 145:
        if len(suffix) > 0:
            design_dir = f'designed_seqs/{DESIGN_TYPE}_minimal_tgt/{suffix}/{model_type}/minimal_{design_seq_len}'
        else:
            design_dir = f'designed_seqs/{DESIGN_TYPE}_minimal_tgt/{model_type}/minimal_{design_seq_len}'
        if not os.path.exists(design_dir):
            os.makedirs(design_dir)
    else:
        if len(suffix) > 0:
            design_dir = f'designed_seqs/{DESIGN_TYPE}_{suffix}/{model_type}'
        else:
            design_dir = f'designed_seqs/{DESIGN_TYPE}/{model_type}'
        if not os.path.exists(design_dir):
            os.makedirs(design_dir)

    model_dir = f'../cf_model_dir/test_fold_0/{model_type}'

    for cell_type_idx, cell_type in enumerate(CELL_TYPES):

        target = targets[cell_type_idx]

        output_filepath = os.path.join(design_dir, f"designed_{model_type}_{DESIGN_TYPE}_{suffix}_cell_type_{cell_type_idx}_{cell_type}_seqs.fasta")
        if os.path.exists(output_filepath):
            print(f"Generating sequences for cell type {cell_type} already exist. Skipping.")
            continue
        
        if cell_type not in cell_types_to_design:
            continue
        print(f"Generating sequences for cell type {cell_type} ({cell_type_idx + 1} / {len(CELL_TYPES)})...")

        # Make model
        # model_ensemble = make_ensemble_model(models_individual, seq_length, output_idx_to_maximize=[cell_type_idx])
        if ~use_maxmin:
            model_ensemble = load_ensemble_model(model_dir,model_basename,range(1,1+n_models_to_ensemble),
                                                design_seq_len,max_seq_len=max_seq_len)
        else:
            model_ensemble = load_maxmin_ensemble_model(model_dir,model_basename,range(1,1+n_models_to_ensemble),
                                                design_seq_len,cell_type,max_seq_len=max_seq_len)

        # # this output maximizes cell type specificity <---- might need to debug the keras backend absolute function
        # def get_target_loss_func(target):

        #     def target_loss_func_h2k_tgt(model_preds):
        #         # model_preds has dimensions (n_seqs, n_outputs)
        #         target_score = K.abs(tensorflow.reduce_mean(model_preds[:, cell_type_idx]-model_preds[:, 1-cell_type_idx]) - target)
        #         return target_score
            
        #     return target_loss_func_h2k_tgt
        
        # can use same loss function for min1, if target is negative
        def get_target_loss_func_max1(target):

            def target_loss_func_max1_tgt(model_preds):
                # model_preds has dimensions (n_seqs, n_outputs)
                target_score = K.abs(tensorflow.reduce_mean(model_preds[:, cell_type_idx]) - target)
                return target_score
            
            return target_loss_func_max1_tgt
    
                
        # Define PWM loss
        # The following is supposed to penalize repeats
        def pwm_loss_func(pwm):
            # PWM has dimensions (n_seqs, seq_length, n_channels)
            # return tensorflow.reduce_mean(pwm[:, :-2, :] * pwm[:, 1:-1, :] * pwm[:, 2:, :])
            return tensorflow.reduce_mean(pwm[:, :-1, :] * pwm[:, 1:, :])
        
        target_loss_func = get_target_loss_func_max1(target)

        seq_vals, pred_vals, train_hist = corefsp.design_seqs(
            model_ensemble,
            target_loss_func,
            pwm_loss_func=pwm_loss_func,
            seq_length=design_seq_len,
            n_seqs=n_seqs_per_target,
            target_weight=1,
            pwm_weight=2.5, # was previously 3
            entropy_weight=1e-3,
            learning_rate=0.001,
            n_iter_max=1000,
            # init_seed=0,
        )

        # get the 0 and 1 indices of pred vals - this is for vgg_shendure models with 3 outputs
        pred_vals = pred_vals[:,:2]

        # sort generated preds descending by generated_preds[:,cell_type_idx] - generated_preds[:,1-cell_type_idx]
        if suffix == 'max1':
            sort_inds = np.argsort(pred_vals[:,cell_type_idx])[::-1]
        elif suffix == 'min1':
            sort_inds = np.argsort(pred_vals[:,cell_type_idx])
        else:
            print("ERROR: suffix must be 'max1' or 'min1'")
            return
        
        seq_vals = seq_vals[sort_inds]
        pred_vals = pred_vals[sort_inds]

        # Save plots
        fig = design_plots.plot_train_history(train_hist)
        fig.savefig(
            os.path.join(design_dir, f"designed_{model_type}_{DESIGN_TYPE}_cell_type_{cell_type_idx}_{cell_type}_{model_type}_tgt{target:.2f}_train_history.png"),
            dpi=200, bbox_inches='tight',
        )
        plt.close(fig)

        fig = design_plots.plot_bar_preds(pred_vals)
        fig.axes[0].set_title(f"Predicted signal, {len(seq_vals):,} seqs designed for {cell_type}")
        fig.savefig(
            os.path.join(design_dir, f"designed__{model_type}_{DESIGN_TYPE}_cell_type_{cell_type_idx}_{cell_type}_{model_type}_tgt{target:.2f}_bar_signal.png"),
            dpi=200, bbox_inches='tight',
        )
        plt.close(fig)

        fig = design_plots.plot_sequence_bitmap(seq_vals)
        fig.savefig(
            os.path.join(design_dir, f"designed_{model_type}_{DESIGN_TYPE}_cell_type_{cell_type_idx}_{cell_type}_tgt{target:.2f}_seq_bitmap.png"),
            dpi=200, bbox_inches='tight',
        )
        plt.close(fig)

        fig = design_plots.plot_n_seqs(seq_vals, n_seqs=10)
        fig.savefig(
            os.path.join(design_dir, f"designed_{model_type}_{DESIGN_TYPE}_cell_type_{cell_type_idx}_{cell_type}_tgt{target:.2f}_seqs.png"),
            dpi=200, bbox_inches='tight',
        )
        plt.close(fig)

        # Save training history data
        train_hist_filepath = os.path.join(design_dir, f"designed_{model_type}_{DESIGN_TYPE}_cell_type_{cell_type_idx}_{cell_type}_tgt{target:.2f}_train_hist.pickle")
        with open(train_hist_filepath, 'wb') as handle:
            pickle.dump(train_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save sequences
        h2k_dir = -1 if cell_type == 'K562' else 1 # use this to flip the sign of the target for K562
        seqs = []
        alphabet = ['A', 'C', 'G', 'T']
        seqs_index = np.argmax(seq_vals, axis=-1)
        for seq_index in seqs_index:
            seqs.append(''.join([alphabet[idx] for idx in seq_index]))
        seq_records = []
        for seq_idx, seq in enumerate(seqs):
            seq_id = f'designed_{model_type}_{DESIGN_TYPE}_{suffix}_{cell_type_idx}_{cell_type}_tgt_{(target*h2k_dir):.2f}_seq_{seq_idx}'
            seq_records.append(Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq), id=seq_id, description=""))
        with open(output_filepath, "a") as output_handle:
            Bio.SeqIO.write(seq_records, output_handle, "fasta")

        print(f"Done with cell type {cell_type}, target {target:.2f}.")

    print("Done.")

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Run Fast SeqProp and generate sequences')
    parser.add_argument('model_type', type=str, help='Model type')
    parser.add_argument('-l','--design_seq_len',type=int,default=145,help='Length of sequences to design')
    parser.add_argument('-v','--valid_idx',type=int,default=None,help='Validation fold index if using validation model')
    parser.add_argument('-t','--target_function',type=str,default='h2k',help='Target function to use')
    parser.add_argument('-s','--suffix',type=str,default='',help='Suffix to add to output directory')
    parser.add_argument('-p','--pretrained',type=bool,default=False,help='Use pretrained model (i.e. skip the design step)')
    # parser.add_argument('-mm','--maxmin',type=bool,default=False,help='Use maxmin model instead of ensemble model')

    args = parser.parse_args()
    model_type = args.model_type
    design_seq_len = args.design_seq_len
    val_idx = args.valid_idx
    target_function = args.target_function
    suffix = args.suffix
    pretrained = args.pretrained
    # use_maxmin = args.maxmin

    # # maximize 1 cell type, not cell type difference
    # if target_function != 'h2k':
    #     suffix = 'max1'
    # if use_maxmin:
    #     print("Using maxmin model instead of ensemble model")

    # target_vec = np.arange()

    model_dir = f'../cf_model_dir/test_fold_0/{model_type}'
    model_basename = MODEL_TYPE_DF.loc[model_type]['model_basename']
    input_len = MODEL_TYPE_DF.loc[model_type]['input_len']
    n_models_to_ensemble = MODEL_TYPE_DF.loc[model_type]['n_ensemble']
    max_pred_h2k = MODEL_TYPE_DF.loc[model_type]['max_pred_h2k']
    max_pred_k2h = MODEL_TYPE_DF.loc[model_type]['max_pred_k2h']
    hepg2_target = max_pred_h2k*1.1
    k562_target = max_pred_k2h*1.1
    targets = [hepg2_target,k562_target]


    use_maxmin = False

    # generate sequences for both cell types!
    if not pretrained:
        generate_sequences_target(model_type,design_seq_len=design_seq_len,target_function=target_function,suffix=suffix,use_maxmin=use_maxmin)

    # Directory depends on target weight
    if design_seq_len < 145:
        if len(suffix) > 0:
            design_dir = f'designed_seqs/{DESIGN_TYPE}_minimal_tgt/{suffix}/{model_type}/minimal_{design_seq_len}'
        else:
            design_dir = f'designed_seqs/{DESIGN_TYPE}_minimal_tgt/{model_type}/minimal_{design_seq_len}'
        if not os.path.exists(design_dir):
            os.makedirs(design_dir)
    else:
        if len(suffix) > 0:
            design_dir = f'designed_seqs/{DESIGN_TYPE}_{suffix}/{model_type}'
        else:
            design_dir = f'designed_seqs/{DESIGN_TYPE}/{model_type}'
        if not os.path.exists(design_dir):
            os.makedirs(design_dir)

    # now, generate: seq_df, combined edit_distance plot, combined pred plot, repeat analysis, (bitmap should already be generated)
    seq_df = pd.DataFrame(columns=['sequence_name','sequence','cell_type','target'])

    # n_lines_per_seq = 3
    FASTA_LINE_LIMIT = 60
    n_lines_per_seq = (design_seq_len // FASTA_LINE_LIMIT) + 1
    for cell_type_idx, cell_type in enumerate(CELL_TYPES):

        seq_file = f'{design_dir}/designed_{model_type}_{DESIGN_TYPE}_{suffix}_cell_type_{cell_type_idx}_{cell_type}_seqs.fasta'

        # initialize
        tgt = -1
        
        # read in fasta file
        with open(seq_file, 'r') as f:
            for line in f:
                if line[0] == '>':
                    seq_name = line[1:].strip()
                    seq = ''
                    line_cnt = 0
                    tgt = float(seq_name.split('_')[-3])
                    # seq_id = f'designed_{model_type}_{DESIGN_TYPE}_target_{cell_type_idx}_{cell_type}_tgt_{target:.2f}_seq_{seq_idx}'
                else:
                    seq += line.strip()
                    line_cnt += 1
                    # seq_df = seq_df.append({'sequence_name':seq_name,'seq':seq}, ignore_index=True)
                    if line_cnt == n_lines_per_seq:
                        seq_df = pd.concat([seq_df, pd.DataFrame({'sequence_name':seq_name,'sequence':seq,'cell_type':cell_type,'target': tgt}, index=[0])], ignore_index=True)


    # seq_df['cell_type'] = seq_df['sequence_name'].apply(lambda x: 'HEPG2' if 'HEPG2' in x else 'K562')

    K.clear_session()
    if val_idx is not None:
        model = load_ensemble_model(model_dir,model_basename,[val_idx],
                                        design_seq_len,max_seq_len=input_len)
    else:                                    
        model = load_ensemble_model(model_dir,model_basename,range(1,1+n_models_to_ensemble),
                                            design_seq_len,max_seq_len=input_len)

    x = np.array([seq_to_one_hot(seq) for seq in seq_df['sequence'].values])
    print(x.shape)
    # # pad x to 500 bp - unnecessary, just load model to do 0-padding at input
    # x = np.pad(x, ((0,0),(0,input_len-x.shape[1]),(0,0)), 'constant', constant_values=0)
    y_hat = model.predict(x)

    # add y_hat to seq_df as columns log2(HEPG2)_pred and log2(K562)_pred
    seq_df['log2(HEPG2)_pred'] = y_hat[:,0]
    seq_df['log2(K562)_pred'] = y_hat[:,1]
    seq_df['log2(H2K)_pred'] = y_hat[:,0] - y_hat[:,1]

    # calculate max repeat length in each sequence
    seq_df['max_repeat'] = seq_df['sequence'].apply(longest_repeat)

    # add design type
    seq_df['design_type'] = f'{DESIGN_TYPE}_{suffix}'

    # sort descending by log2(H2K)_pred and reset index
    seq_df = seq_df.sort_values(by='log2(H2K)_pred', ascending=False).reset_index(drop=True)

    # save seq_df
    seq_df.to_csv(f'{design_dir}/designed_seqs_{model_type}_{DESIGN_TYPE}_seq_df.csv', index=False)


    # plot bar plot of max_repeat value counts in seq_df
    fig,ax = plt.subplots(figsize=(3,3))
    sns.countplot(x='max_repeat', data=seq_df,color='tab:blue',ax=ax)
    ax.set_xlabel('Max repeat length')
    ax.set_ylabel('Count')
    ax.set_title(f'{model_type} {DESIGN_TYPE} max repeats')
    fig.savefig(
        f"{design_dir}/designed_{model_type}_{DESIGN_TYPE}_max_repeat_cntplot.png",
        dpi=120,
        bbox_inches='tight',
    )
    plt.close(fig)


    # histogram of log2(HEPG2)_pred
    fig,ax = plt.subplots(figsize=(5,3))
    # plt.hist(seq_df['log2(H2K)_pred'], bins=300)
    # plot histogram colored by cell type
    sns.histplot(data=seq_df, x='log2(H2K)_pred', hue='cell_type', palette=PALETTE, bins=50,element='step',fill=True,ax=ax)
    # plot vertical line at MAX_PRED_H2K_T0_V1
    plt.axvline(x=max_pred_h2k, color='k', linestyle='--')
    # plot vertical line at MAX_PRED_K2H_T0_V2
    plt.axvline(x=max_pred_k2h, color='k', linestyle='--')
    ax.set_xlabel('pred. log2(HEPG2/K562)')
    ax.set_ylabel('count')
    ax.set_title(f'{model_type} {DESIGN_TYPE} designed seq preds')
    # hide legend
    ax.legend().remove()
    # add grid lines
    plt.grid(axis='y',zorder=-1)

    # add median line
    plt.axvline(x=np.median(seq_df[seq_df['cell_type']=='HEPG2']['log2(H2K)_pred']), color='tab:orange', linestyle='--')
    plt.axvline(x=np.median(seq_df[seq_df['cell_type']=='K562']['log2(H2K)_pred']), color='tab:blue', linestyle='--')

    # add plots with target lines
    plt.axvline(x=hepg2_target, color='tab:orange', linestyle=':')
    plt.axvline(x=k562_target, color='tab:blue', linestyle=':')

    fig.savefig(
        f"{design_dir}/designed_{model_type}_{DESIGN_TYPE}_pred_hist.png",
        # 'temp.png',
        dpi=120,
        bbox_inches='tight',
    )
    plt.close(fig)


    # Plot with edit distances
    print(f"DEBUG: {seq_df[seq_df['cell_type']=='HEPG2']['sequence'].values.shape}")
    hepg2_distances = get_paired_editdistances(seq_df[seq_df['cell_type']=='HEPG2']['sequence'].values)
    k562_distances = get_paired_editdistances(seq_df[seq_df['cell_type']=='K562']['sequence'].values)
    # make temp_df with extra column for cell type
    temp_df = pd.DataFrame({'edit_distance':np.concatenate([hepg2_distances,k562_distances]),
                        'cell_type':np.concatenate([['HEPG2']*len(hepg2_distances),['K562']*len(k562_distances)])})

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    sns.violinplot(x='cell_type', y='edit_distance', data=temp_df, ax=ax, palette=PALETTE)
    # ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Edit distance / nucleotide\n')
    ax.set_xlabel('Target Cell Type')

    # add '{:.3f} +/- {:.3f}'.format(np.mean(distances), np.std(distances) as text for each cell type
    ax.text(0, 0.9, '{:.3f} +/- {:.3f}'.format(np.mean(hepg2_distances), np.std(hepg2_distances)), ha='center', va='center', color='tab:orange')
    ax.text(1, 0.9, '{:.3f} +/- {:.3f}'.format(np.mean(k562_distances), np.std(k562_distances)), ha='center', va='center', color='tab:blue')
    # gridlines
    plt.grid(axis='y',zorder=-1)
    ax.set_title(f'{model_type} {DESIGN_TYPE} edit distances')

    fig.savefig(
        f"{design_dir}/designed_{model_type}_{DESIGN_TYPE}_editdistance.png",
        dpi=120,
        bbox_inches='tight',
    )
    plt.close(fig)