import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import comb
import sys
import os
import argparse

from scipy.stats import pearsonr, spearmanr, mannwhitneyu

### Define functions ###

cluster_map_fname = "motif_proc_aux_files/jaspar_motif_id_to_cluster.csv"

def load_fimo_df(fname,qthresh=0.03,cast_seqname_to_int=True,pthresh=None):

    # load fimo results tsv
    fimo_df = pd.read_csv(fname, sep='\t')
    # remove last three rows (which are text output of script, not data)
    fimo_df = fimo_df[:-3]
    # filter on p-value if pthresh supplied, otherwise filter on q-value
    if pthresh is not None:
        fimo_df = fimo_df[fimo_df['p-value'] < pthresh]
    else:
        # filter for q-value
        fimo_df = fimo_df[fimo_df['q-value'] < qthresh]
    # set sequence name, start, stop types to int
    if cast_seqname_to_int:
        fimo_df['sequence_name'] = fimo_df['sequence_name'].astype(int)
    fimo_df['start'] = fimo_df['start'].astype(int)-1 # the -1 is to convert from 1-indexed to 0-indexed
    fimo_df['stop'] = fimo_df['stop'].astype(int) # by not subtracting 1 convert stop from inclusive to exclusive

    # NEW! Add cluster column for subsequent cluster collapsing (rather than motif_id collapsing)
    motif_id_to_cluster_df = pd.read_csv(cluster_map_fname) # should make this an argument at some point
    motif_id_to_cluster_df.rename(columns={'cluster_id':'jaspar_cluster'},inplace=True)

    # to final_df add a column with the motif cluster
    fimo_df = fimo_df.merge(motif_id_to_cluster_df, on='motif_id', how='left')

    return fimo_df

# allow 3 bp overlap between hits max 
def check_overlap(start1, stop1, start2, stop2,overlap=3):
    return (start1 <= start2+overlap <= stop1) or (start1 <= stop2-overlap <= stop1) or \
            (start2 <= start1+overlap <= stop2) or (start2 <= stop1-overlap <= stop2)

def cluster_motifs_in_fimo_df(fname,qthresh=0.03,overlap=3,by_score=False,cast_seqname_to_int=False,pthresh=None,col_to_cluster='motif_id'):

    fimo_df = load_fimo_df(fname,qthresh=qthresh,cast_seqname_to_int=cast_seqname_to_int,pthresh=pthresh)
    seqs_with_hits = fimo_df['sequence_name'].unique()

    n_seqs_with_hits = len(seqs_with_hits)

    for seq_idx,seq in enumerate(seqs_with_hits):
        # if at a 10% increment, print progress
        if seq_idx % (n_seqs_with_hits//10) == 0:
            print(f'{seq_idx}/{n_seqs_with_hits}')
        cur_motif_df = fimo_df[fimo_df['sequence_name']==seq]
        cur_motif_df = cur_motif_df.sort_values(by='p-value') # this should already be true? but just in case
        motifs = cur_motif_df[col_to_cluster].unique()
        
        # for each motif, cluster overlapping hits together
        for motif in motifs:
            cur_motif_hits = cur_motif_df[cur_motif_df[col_to_cluster]==motif]
            cur_motif_hits = cur_motif_hits.sort_values(by='start')
            cur_motif_hits = cur_motif_hits.reset_index(drop=True)
            
            # cluster overlapping hits together
            overlapping_lists = [] # will contain pairs of overlapping hits
            # mapped_inds = []
            for i in range(cur_motif_hits.shape[0]):
                # if i in mapped_inds: continue
                # mapped_inds.append(i)
                overlapping_lists.append([i]) # this accounts for when there is only 1 motif hit in this sequence
                for j in range(i+1, cur_motif_hits.shape[0]):
                    # if j in mapped_inds: continue
                    # if check_overlap(cur_motif_hits.loc[i,'start'], cur_motif_hits.loc[i,'midpt'], cur_motif_hits.loc[i,'stop'],
                    #                 cur_motif_hits.loc[j,'start'], cur_motif_hits.loc[j,'midpt'], cur_motif_hits.loc[j,'stop']):
                    if check_overlap(cur_motif_hits.loc[i,'start'], cur_motif_hits.loc[i,'stop'],
                                    cur_motif_hits.loc[j,'start'], cur_motif_hits.loc[j,'stop'],overlap=overlap):
                        overlapping_lists.append([i,j])
                        # mapped_inds.append(j)
            
            # merge overlapping lists
            for i in range(len(overlapping_lists)):
                for j in range(i+1, len(overlapping_lists)):
                    if set(overlapping_lists[i]).intersection(set(overlapping_lists[j])):
                        overlapping_lists[i] = list(set(overlapping_lists[i]).union(set(overlapping_lists[j])))
                        overlapping_lists[j] = []
            overlapping_lists = [x for x in overlapping_lists if x != []]
            
            # for each discrete hit cluster, select the hit with the lowest p-value as the "representative" hit, and save correspond row to new df
            cur_motif_hits['representative'] = False
            for hit_cluster in overlapping_lists:
                # cur_motif_hits.loc[hit_cluster[0],'representative'] = True
                if by_score:
                    cur_motif_hits.loc[cur_motif_hits.loc[hit_cluster,'score'].idxmax(),'representative'] = True
                else:
                    cur_motif_hits.loc[cur_motif_hits.loc[hit_cluster,'p-value'].idxmin(),'representative'] = True
            cur_motif_hits = cur_motif_hits[cur_motif_hits['representative']==True]
            cur_motif_hits = cur_motif_hits.drop(columns=['representative'])
            
            # add cur_motif_hits to new df
            if 'clustered_motif_df' not in locals():
                clustered_motif_df = cur_motif_hits
            else:
                clustered_motif_df = pd.concat([clustered_motif_df, cur_motif_hits])

    clustered_motif_df = clustered_motif_df.sort_values(by='p-value')
    clustered_motif_df = clustered_motif_df.reset_index(drop=True)
    return clustered_motif_df

def cluster_motifs_by_pos(clustered_motif_df,overlap=3,by_score=False):

    seqs_with_hits = clustered_motif_df['sequence_name'].unique()
    n_seqs_with_hits = len(seqs_with_hits)

    for seq_idx,seq in enumerate(seqs_with_hits):
        # if at a 10% increment, print progress
        if seq_idx % (n_seqs_with_hits//10) == 0:
            print(f'{seq_idx}/{n_seqs_with_hits}')

        cur_motif_cluster_df = clustered_motif_df[clustered_motif_df['sequence_name']==seq]
        cur_motif_cluster_df = cur_motif_cluster_df.sort_values(by='start')
        cur_motif_cluster_df = cur_motif_cluster_df.reset_index(drop=True)

        # step through every position in the sequence (up to the final motif's start coordinate, which will save a tiny bit of time maybe)
        unique_pos_hits = []
        for i in range(cur_motif_cluster_df['start'].max()+1):
            # get all hits with start == i
            cur_hits = cur_motif_cluster_df[cur_motif_cluster_df['start']==i]
            # select hit with the lowest p-value
            if cur_hits.shape[0] > 0:
                cur_hit = cur_hits.loc[cur_hits['p-value'].idxmin()]
                if len(unique_pos_hits) == 0:
                    unique_pos_hits.append(cur_hit)
                else:
                    # check if cur_hit overlaps with any hits in unique_pos_hits
                    overlap_flag = False
                    for j in range(len(unique_pos_hits)):
                        if check_overlap(unique_pos_hits[j]['start'], unique_pos_hits[j]['stop'], cur_hit['start'], cur_hit['stop'],overlap=overlap):
                            overlap_flag = True
                            # check if cur_hit has a lower p-value than the hit in unique_pos_hits
                            if cur_hit['p-value'] < unique_pos_hits[j]['p-value']:
                                # if so, replace the hit in unique_pos_hits with cur_hit
                                unique_pos_hits[j] = cur_hit
                                # Note: it's possible for a motif to swallow multiple previous hits, so need to remove duplicates at the end
                    if not overlap_flag:
                        unique_pos_hits.append(cur_hit)

        pos_clustered_df = pd.DataFrame(unique_pos_hits)
        pos_clustered_df = pos_clustered_df.sort_values(by='start')
        pos_clustered_df = pos_clustered_df.reset_index(drop=True)
        # remove duplicates
        pos_clustered_df = pos_clustered_df.drop_duplicates(subset=['motif_id','start','stop'])

        # add cur_motif_cluster_df to new df
        if 'final_df' not in locals():
            final_df = pos_clustered_df
        else:
            final_df = pd.concat([final_df, pos_clustered_df])

    final_df = final_df.reset_index(drop=True)
    final_df['motif_alt_id'] = final_df['motif_alt_id'].str.upper()
    
    return final_df

