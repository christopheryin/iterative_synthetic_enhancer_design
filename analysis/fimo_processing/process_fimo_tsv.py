import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import comb
import sys
import os
import argparse

from scipy.stats import pearsonr, spearmanr, mannwhitneyu
# from design_utils import MODEL_TYPE_DF

from fimo_proc_utils import cluster_motifs_in_fimo_df, cluster_motifs_by_pos, check_overlap

matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

# from analyze_motifs import *


### extract arguments from command line ###

parser = argparse.ArgumentParser()
parser.add_argument('model_type')
parser.add_argument('design_type')
# parser.add_argument("fimo_fname", help="path to fimo.tsv file")
# parser.add_argument("seq_fname", help="path to sequences + measurement dataframe file") # must contain sequence name and measurement columns
parser.add_argument("-q","--qthresh", default=0.05,help="q-value threshold for clustering")
parser.add_argument("-o","--output_dir", default="saved_processed_motif_files",help="directory to save processed motif files")
parser.add_argument("-cm","--cluster_map_fname", default="motif_proc_aux_files/jaspar_motif_id_to_cluster.csv",help="filename to save cluster map to")
parser.add_argument("-l",'--len',default=None,type=int,help="length of minimal enhancer")

args = parser.parse_args()
model_type = args.model_type
design_type = args.design_type
# fname = args.fimo_fname
# seq_fname = args.seq_fname
qthresh = args.qthresh
output_dir = args.output_dir
cluster_map_fname = args.cluster_map_fname
seq_len = args.len

basename = f'{model_type}_{design_type}'
if seq_len is None:
    fname = f'designed_seqs/{design_type}/{model_type}/{model_type}_{design_type}_fimo.tsv'
    seq_fname = f'designed_seqs/{design_type}/{model_type}/designed_seqs_{model_type}_{design_type}_seq_df.csv'
    output_dir = f'designed_seqs/{design_type}/{model_type}'
else:
    fname = f'designed_seqs/{design_type}/{model_type}/minimal_{seq_len}/{model_type}_{design_type}_{seq_len}_fimo.tsv'
    seq_fname = f'designed_seqs/{design_type}/{model_type}/minimal_{seq_len}/designed_seqs_{model_type}_{design_type}_{seq_len}_seq_df.csv'
    output_dir = f'designed_seqs/{design_type}/{model_type}/minimal_{seq_len}'

# extract fname basename minus the .tsv
basename = os.path.basename(fname).split('.')[0]

pthresh = None

cluster_map_fname = "motif_proc_aux_files/jaspar_motif_id_to_cluster.csv"

# create output dir if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

thresh_suffix = f'{pthresh:.0e}' if pthresh is not None else f'{qthresh:.2f}'
clustered_df_fname = f'{basename}_clustered_motif_df_{"p" if pthresh is not None else "q"}thresh{thresh_suffix}.csv'
final_df_fname = f'{basename}_final_df_{"p" if pthresh is not None else "q"}thresh{thresh_suffix}.csv'
seq_basename = os.path.basename(seq_fname).split('.')[0]
seq_df_fname = f'{seq_basename}_plus_cluster_counts_{"p" if pthresh is not None else "q"}thresh{thresh_suffix}.csv'


### Begin clustering motifs ###

col_to_cluster = 'motif_id' # previously was motif_alt_id, really should have been motif_id but oh well; note if set to jaspar_cluster runtime is really long because that increases per-motif count

print('Clustering motifs by id...')
id_clustered_motif_df = cluster_motifs_in_fimo_df(fname,qthresh=qthresh,pthresh=pthresh,col_to_cluster=col_to_cluster)
print('Done.')
print('Clustering motifs by position...')
pos_clustered_df = cluster_motifs_by_pos(id_clustered_motif_df)
print('Done.')

# save clustered_motif_df and final_df
id_clustered_motif_df.to_csv(f'{output_dir}/{clustered_df_fname}',index=False)
pos_clustered_df.to_csv(f'{output_dir}/{final_df_fname}',index=False)

### Generate seq x cluster_count matrix ###

# load seq_df - should have sequence_name column at minimum, can have other sequence attribute columns (e.g. pred log2(H2K))
seq_df = pd.read_csv(seq_fname)

seq_df['n_motifs'] = 0
for seq_idx,seq_name in enumerate(seq_df['sequence_name']):
    if seq_name in pos_clustered_df['sequence_name'].values:
        seq_df.loc[seq_idx,'n_motifs'] = pos_clustered_df[pos_clustered_df['sequence_name']==seq_name].shape[0]

# create new numpy array that's n_seqs x n_clusters
seq_cluster_matrix = np.zeros((seq_df.shape[0], pos_clustered_df['jaspar_cluster'].nunique()))
print('Computing sequence x cluster count matrix...')
# fill in the matrix with the number of motifs in each cluster for each sequence
for seq_idx,seq_name in enumerate(seq_df['sequence_name']):
    for cluster_idx, cluster in enumerate(pos_clustered_df['jaspar_cluster'].unique()):
        seq_cluster_matrix[seq_idx,cluster_idx] = len(pos_clustered_df[(pos_clustered_df['jaspar_cluster']==cluster) & (pos_clustered_df['sequence_name']==seq_name)])
print('Done.')

# create a dataframe from the matrix
seq_cluster_df = pd.DataFrame(seq_cluster_matrix, columns=pos_clustered_df['jaspar_cluster'].unique())

seq_cluster_df['n_unique_clusters'] = seq_cluster_df.apply(lambda x: len(x[x>0]), axis=1)

clusters = pos_clustered_df['jaspar_cluster'].unique()

# append d2_cluster_df to seq_df
seq_df = pd.concat([seq_df, seq_cluster_df], axis=1)
# save
seq_df.to_csv(f'{output_dir}/{seq_df_fname}', index=False)