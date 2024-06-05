import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import corefsp

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import matplotlib as mpl
import argparse
import Bio
import Bio.Seq
import Bio.SeqRecord
import Bio.SeqIO
import Bio.SeqUtils

import os
import sys

sys.path.append('../')
from design_utils import TEST_FOLD_IDX,DESIGNED_SEQ_LEN,MODEL_TYPE_DF,CELL_TYPES, load_ensemble_model, seq_to_one_hot, one_hot_to_seq, longest_repeat

###########################################

nt_color_dict = {
    'A': (15/255, 148/255, 71/255),
    'C': (35/255, 92/255, 153/255),
    'G': (245/255, 179/255, 40/255),
    'T': (212/255, 38/255, 56/255),
}

def plot_sequence_bitmap(seq_vals):
    # Convert sequences to numerical indices
    if type(seq_vals[0])==np.ndarray:
        # Assume one hot-encoded
        seqs_as_index = np.argmax(seq_vals, axis=-1)
    elif type(seq_vals[0])==str:
        # Assume list of strings
        seqs_as_index = [[["A", "C", "G", "T"].index(c) for c in si] for si in seq_vals]
        seqs_as_index = np.array(seqs_as_index)
    else:
        raise ValueError(f"type of seq_vals {type(seq_vals)} not recognized")
            
    # Define colors and colormap
    nt_colors = [nt_color_dict[n] for n in ['A', 'C', 'G', 'T']]
    cmap = mpl.colors.ListedColormap(nt_colors)
    bounds=[0, 1, 2, 3, 4]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Actually plot - seqs_as_index.shape[1] (number of bps), seqs_as_index.shape[0] (number of sequences)
    fig, ax = plt.subplots(figsize=(0.03*seqs_as_index.shape[1], 0.03*seqs_as_index.shape[0]))
    ax.imshow(
        seqs_as_index[::-1] + 0.5,
        aspect='equal',
        interpolation='nearest',
        origin='lower',
        cmap=cmap,
        norm=norm,
    )
    # Custom legend with nucleotide colors
    legend_elements = [
        mpl.patches.Patch(facecolor=c, label=nt)
        for nt, c in nt_color_dict.items()
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.7, 1.025), fontsize='medium')

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel('Position (nt)')
    ax.set_ylabel('Generated sequences')

    return fig

def load_d2_deseq_df():

    log2fc_df = pd.read_csv(f'd2_data/chris_log2fc_df_clean2.csv')

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

def load_motif_data_d2(qthresh_suffix=''):
    d2_final_df = pd.read_csv(f'd2_data/d2_final_df{qthresh_suffix}_v2.csv')
    d2_clustered_motif_df = pd.read_csv(f'd2_data/d2_clustered_motif_df{qthresh_suffix}.csv')
    d2_deseq_plus_cluster_cnts_df = pd.read_csv(f'd2_data/d2_deseq_plus_cluster_cnts_df{qthresh_suffix}_v2.csv')

    # For all HEPG2 Control and K562 control sequences, change model name to control_f if "f_" is in the name, and control_r if "r_" is in the name
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="HEPG2_CTRL",'model'] = d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="HEPG2_CTRL",'model'].apply(lambda x: "control_f" if "f_" in x else "control_r")
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="K562_CTRL",'model'] = d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="K562_CTRL",'model'].apply(lambda x: "control_f" if "f_" in x else "control_r")
    # now change cell_type to HEPG2 if HEPG2 Control, and K562 if K562 Control
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="HEPG2_CTRL",'cell_type'] = "HEPG2"
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['cell_type']=="K562_CTRL",'cell_type'] = "K562"

    # if d2_deseq_df['model'] contains crafted, change 'generator' to 'motif_repeat'
    d2_deseq_plus_cluster_cnts_df.loc[d2_deseq_plus_cluster_cnts_df['model'].str.contains("crafted"),'generator'] = "motif_repeat"

    return d2_final_df, d2_clustered_motif_df, d2_deseq_plus_cluster_cnts_df

def target_loss_func_h2k(y_pred):
    return -tf.reduce_sum(y_pred[:,0]-y_pred[:,1])

def target_loss_func_k2h(y_pred):
    return tf.reduce_sum(y_pred[:,0]-y_pred[:,1])

# penalize repeats
def pwm_loss_func(pwm):
    # PWM has dimensions (n_seqs, seq_length, n_channels)
    # return tensorflow.reduce_mean(pwm[:, :-2, :] * pwm[:, 1:-1, :] * pwm[:, 2:, :])
    return tensorflow.reduce_mean(pwm[:, :-1, :] * pwm[:, 1:, :])

DESIGN_SEQ_LENGTH = 145
DESIGN_TYPE = 'nonmotif_masked_fsp'
N_SEQS_PER_CELL_TYPE = 5
N_VARS_PER_SEQ = 10 # generate 50 but keep 10, so generate 10*10


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Train a DEN and generate sequences for a given cell type.')
    # parser.add_argument('cell_type_idx', type=int, help='Cell type index')
    parser.add_argument('model_type', type=str, help='Model type')
    parser.add_argument('cell_type', type=str, help='Cell type')
    parser.add_argument('-d','--direction', type=str, default='h2k', help='Direction of perturbation (h2k or k2h)')
    parser.add_argument('-v','--valid_idx', type=int, default=None, help='Validation model')
    args = parser.parse_args()

    model_type = args.model_type
    cell_type = args.cell_type
    direction = args.direction
    valid_idx = args.valid_idx

    model_dir = f'../cf_model_dir/test_fold_0/{model_type}'
    model_basename = MODEL_TYPE_DF.loc[model_type]['model_basename']
    input_len = MODEL_TYPE_DF.loc[model_type]['input_len']
    n_models_to_ensemble = MODEL_TYPE_DF.loc[model_type]['n_ensemble']
    max_pred_h2k = MODEL_TYPE_DF.loc[model_type]['max_pred_h2k']
    max_pred_k2h = MODEL_TYPE_DF.loc[model_type]['max_pred_k2h']

    # create directory for saving outputs
    if not os.path.exists(f"designed_seqs/{DESIGN_TYPE}"):
        os.makedirs(f"designed_seqs/{DESIGN_TYPE}")
    if not os.path.exists(f"designed_seqs/{DESIGN_TYPE}/{model_type}"):
        os.makedirs(f"designed_seqs/{DESIGN_TYPE}/{model_type}")

    output_dir = f'designed_seqs/{DESIGN_TYPE}/{model_type}'

    K.clear_session()
    # padded_model
    design_model = load_ensemble_model(model_dir,model_basename,range(1,1+n_models_to_ensemble),
                                        DESIGN_SEQ_LENGTH,max_seq_len=input_len)

    # load d2_deseq_df
    d2_deseq_df = load_d2_deseq_df()

    d2_final_df,_,d2_deseq_df = load_motif_data_d2(qthresh_suffix='_qthresh05')
    # rebase d2_final_df start and stop to 0 indexed
    d2_final_df['start'] -= 1
    # d2_final_df['stop'] -= 1 # don't adjust stop; this is the last base that is included in the sequence

    # seq_df = pd.DataFrame(columns=['sequence_name','design_type','cell_type','sequence'])

    # test out for a single sequence the motif perturbation options

    motif_flank_buffer = 0 # how many bps to each side of motif to include in motif

    d2_deseq_df = d2_deseq_df[d2_deseq_df['cell_type']==cell_type]

    # sort d2_deseq_df by descending log2FoldChange_H2K_deseq if cell_type == HEPG2, else ascending
    d2_deseq_df = d2_deseq_df.sort_values(by='log2FoldChange_H2K_deseq', ascending= cell_type=='K562')

    # for the top N_SEQS_PER_CELL_TYPE sequences, design N_VARS_PER_SEQ*10 sequences optimizing the nonmotif sequence
    for cur_idx in range(N_SEQS_PER_CELL_TYPE):
        # cur_idx = 0 # this is the rank of the sequence in d2_deseq_df by log2(HEPG2/K562)...I am going to want to loop this over n_top_seqs I think...
        seq_idx = d2_deseq_df.index[cur_idx]
        cur_cell_type = d2_deseq_df.iloc[cur_idx]['cell_type']

        cur_seq = d2_deseq_df.iloc[cur_idx]['enhancer']
        cur_seq_motifs = d2_final_df[d2_final_df['sequence_name'] == seq_idx]

        # create mask with motif positions
        motif_mask = np.zeros(DESIGN_SEQ_LENGTH,dtype=int)
        for motif_start, motif_end in zip(cur_seq_motifs['start'].values, cur_seq_motifs['stop'].values) :
            motif_mask[motif_start - motif_flank_buffer:motif_end + motif_flank_buffer] = 1

        # get nonmotif_mask - this can be used as input to masked SeqProp for optimized/adversarial perturbations!
        nonmotif_mask = (1 - motif_mask)

        mask = np.tile(nonmotif_mask,(4,1)).T
        cur_seq_onehot = seq_to_one_hot(cur_seq)
        pattern = np.zeros((DESIGN_SEQ_LENGTH,4)) + cur_seq_onehot * motif_mask[:,None]

        # select target loss function based on which direction to optimize
        target_loss_func = target_loss_func_h2k if direction=='h2k' else target_loss_func_k2h
        cell_type_idx = 0 if cell_type == 'HEPG2' else 1

        seq_vals, pred_vals, train_history = corefsp.design_seqs(
            design_model,
            target_loss_func,
            pwm_loss_func=pwm_loss_func,
            seq_length=DESIGN_SEQ_LENGTH,
            n_seqs=N_VARS_PER_SEQ*5,
            target_weight=1,
            pwm_weight=3,
            entropy_weight=1e-3,
            learning_rate=0.001,
            n_iter_max=500,
            mask=mask,
            pattern=pattern
            # init_seed=0,
        )

        # val model
        # hardcoding validation model for now as dhs64_finetuned_t0_v2
        if valid_idx is not None:
            K.clear_session()
            val_model = load_ensemble_model(model_dir,model_basename,[valid_idx],
                                                DESIGN_SEQ_LENGTH,max_seq_len=input_len)
        else:
            val_model = design_model
        
        x_d2_tot = np.array([seq_to_one_hot(seq) for seq in d2_deseq_df['enhancer']])
        y_hat_d2_tot = design_model.predict(x_d2_tot)
        y_hat_d2_tot_h2k = y_hat_d2_tot[:, 0] - y_hat_d2_tot[:, 1]
        max_pred_h2k = np.max(y_hat_d2_tot_h2k)
        max_pred_k2h = np.max(-y_hat_d2_tot_h2k)

        # subplot 3 columns x 1 row
        fig, ax = plt.subplots(1,3,figsize=(11,3))

        # plot loss
        ax[0].plot(train_history['loss'])
        ax[0].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[0].set_title(f'Training loss')

        # plot val model predictions of H2K on ax[1]
        y_hat_val = val_model.predict(seq_vals)
        ax[1].hist(y_hat_val[:,0]-y_hat_val[:,1],bins=20)
        ax[1].set_xlabel('val log2(HEPG2/K562)')
        ax[1].set_ylabel('count')
        ax[1].set_title(f'Design predictions')

        x_og = np.expand_dims(seq_to_one_hot(cur_seq),axis=0)
        y_og = val_model.predict(x_og)
        print(y_og)
        print(y_og[:,0]-y_og[:,1])

        # plot vertical dashed line at y_og
        ax[1].axvline(y_og[:,0]-y_og[:,1],color='k',linestyle='--')
        # plot vertical dashed line at max_pred_h2k
        ax[1].axvline(max_pred_h2k,color='r',linestyle='--')

        # # plot val model predictions of H2K on ax[1]
        # y_hat_val = val_model.predict(seq_vals)
        # ax[1].hist(y_hat_val[:,0]-y_hat_val[:,1],bins=20)
        # ax[1].set_xlabel('val log2(HEPG2/K562)')
        # ax[1].set_ylabel('count')
        # ax[1].set_title(f'Val predictions')

        # plot train vs val model predictions on ax[2]
        print(type(pred_vals))
        print(pred_vals.shape)
        y_hat_train = pred_vals #reg_model.predict(seq_vals)
        # get spearman and pearson correlations between y_hat_val and y_hat_train
        from scipy.stats import spearmanr, pearsonr
        rs = spearmanr(y_hat_val[:,0]-y_hat_val[:,1],y_hat_train[:,0]-y_hat_train[:,1])[0]
        r2 = pearsonr(y_hat_val[:,0]-y_hat_val[:,1],y_hat_train[:,0]-y_hat_train[:,1])[0]**2
        print(f'Spearman R: {rs:.3f}')
        print(f'R^2       : {r2:.3f}')

        # scatterplot of y_hat_val vs y_hat_train
        ax[2].scatter(y_hat_train[:,0]-y_hat_train[:,1],y_hat_val[:,0]-y_hat_val[:,1])
        ax[2].set_xlabel('train log2(HEPG2/K562)')
        ax[2].set_ylabel('val log2(HEPG2/K562)')
        ax[2].set_title(f'Spearman R: {rs:.3f}, R^2: {r2:.3f}')

        # sort by y_hat_val descending by h2k or k2h
        inds = np.argsort(y_hat_val[:,cell_type_idx]-y_hat_val[:,1-cell_type_idx])[::-1]
        seq_vals = seq_vals[inds]
        y_hat_val = y_hat_val[inds]
        # y_hat_train = y_hat_train[inds]
        y_hat_train = y_hat_val

        fig.savefig(
            f"{output_dir}/designed_{model_type}_{DESIGN_TYPE}_seq{seq_idx}_{direction}_metrics.png",
            dpi=200, bbox_inches='tight',
        )
        plt.close(fig)

        # # plot top 10 sequences
        # for i in range(10):
        #     print(f'{one_hot_to_seq(seq_vals[i])} | {y_hat_val[i,0]-y_hat_val[i,1]:.3f} | {y_hat_train[i,0]-y_hat_train[i,1]:.3f}')
            # print(f'{one_hot_to_seq(seq_vals[i])[insert_pos-n_bps_to_optimize:insert_pos+len(tp53_motif)+n_bps_to_optimize]} | {y_hat_val[i,0]-y_hat_val[i,1]:.3f} | {y_hat_train[i,0]-y_hat_train[i,1]:.3f}')

        fig = plot_sequence_bitmap(seq_vals)
        fig.savefig(
            f"{output_dir}/designed_{model_type}_{DESIGN_TYPE}_seq{seq_idx}_{direction}_bitmap.png",
            dpi=200, bbox_inches='tight',
        )
        plt.close(fig)

        # Save sequences - alt version
        designed_seqs_filepath = f'{output_dir}/designed_{model_type}_{DESIGN_TYPE}_seq{seq_idx}_{direction}.fasta'
        seqs = []
        alphabet = ['A', 'C', 'G', 'T']
        seqs_index = np.argmax(seq_vals, axis=-1) # I think seq_vals is equivalent to generated_onehot, I guess we'll see
        for seq_index in seqs_index:
            seqs.append(''.join([alphabet[idx] for idx in seq_index]))
        seq_records = []
        for idx, seq in enumerate(seqs):
            seq_id = f'designed_{model_type}_{DESIGN_TYPE}_seq{seq_idx}_{direction}_{idx}'
            seq_records.append(Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq), id=seq_id, description=""))
        with open(designed_seqs_filepath, "w") as output_handle:
            Bio.SeqIO.write(seq_records, output_handle, "fasta")