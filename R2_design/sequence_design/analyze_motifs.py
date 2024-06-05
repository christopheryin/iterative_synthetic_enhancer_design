import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import pearsonr, spearmanr, mannwhitneyu

matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

### Functions for loading and collapsing motif hits ###

def load_fimo_df(fname,qthresh=0.03):

    # load fimo results tsv
    fimo_df = pd.read_csv(fname, sep='\t')
    # remove last three rows (which are text output of script, not data)
    fimo_df = fimo_df[:-3]
    # filter for q-value < 0.05
    fimo_df = fimo_df[fimo_df['q-value'] < qthresh]
    # set sequence name, start, stop types to int
    # fimo_df['sequence_name'] = fimo_df['sequence_name'].astype(int)
    fimo_df['start'] = fimo_df['start'].astype(int)
    fimo_df['stop'] = fimo_df['stop'].astype(int)
    # calculate approximate midpoint (can be non-integer)
    fimo_df['midpt'] = fimo_df['start'] + (fimo_df['stop'] - fimo_df['start'])/2

    return fimo_df

# function that takes a pair of start and stop coords and returns true if they overlap
# def check_overlap(start1, stop1, start2, stop2):
#     return (start1 <= start2 <= stop1) or (start1 <= stop2 <= stop1) or (start2 <= start1 <= stop2) or (start2 <= stop1 <= stop2)

# def check_overlap(start1, midpt1, stop1, start2, midpt2, stop2, overlap=2):
#     return (start1 <= midpt2 <= stop1 + overlap) or (start1 - overlap <= midpt2 <= stop1) or \
#             (start2 <= midpt1 <= stop2 + overlap) or (start2 - overlap <= midpt1 <= stop2)

# allow 3 bp overlap between hits max 
def check_overlap(start1, stop1, start2, stop2,overlap=3):
    return (start1 <= start2+overlap <= stop1) or (start1 <= stop2-overlap <= stop1) or \
            (start2 <= start1+overlap <= stop2) or (start2 <= stop1-overlap <= stop2)

def cluster_motifs_in_fimo_df(fname,qthresh=0.03,overlap=3):

    fimo_df = load_fimo_df(fname,qthresh=0.03)
    seqs_with_hits = fimo_df['sequence_name'].unique()

    for seq in seqs_with_hits:
        cur_motif_df = fimo_df[fimo_df['sequence_name']==seq]
        cur_motif_df = cur_motif_df.sort_values(by='p-value') # this should already be true? but just in case
        motifs = cur_motif_df['motif_alt_id'].unique()
        
        # for each motif, cluster overlapping hits together
        for motif in motifs:
            cur_motif_hits = cur_motif_df[cur_motif_df['motif_alt_id']==motif]
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

def cluster_motifs_by_pos(clustered_motif_df,overlap=3):

    seqs_with_hits = clustered_motif_df['sequence_name'].unique()

    for seq in seqs_with_hits:
        cur_motif_cluster_df = clustered_motif_df[clustered_motif_df['sequence_name']==seq]
        cur_motif_cluster_df = cur_motif_cluster_df.sort_values(by='start')
        cur_motif_cluster_df = cur_motif_cluster_df.reset_index(drop=True)

        # cluster overlapping hits together
        overlapping_lists = [] # will contain pairs of overlapping hits
        # mapped_inds = []

        n_cur_motif_cluster = cur_motif_cluster_df.shape[0]
        for i in range(n_cur_motif_cluster):
            # if i in mapped_inds: continue
            # mapped_inds.append(i)
            overlapping_lists.append([i]) # this accounts for the edge case where there is only 1 motif in the cluster
            for j in range(i+1, n_cur_motif_cluster):
                # if j in mapped_inds: continue
                # if check_overlap(cur_motif_cluster_df.loc[i,'start'], cur_motif_cluster_df.loc[i,'midpt'], cur_motif_cluster_df.loc[i,'stop'],
                #                 cur_motif_cluster_df.loc[j,'start'], cur_motif_cluster_df.loc[j,'midpt'], cur_motif_cluster_df.loc[j,'stop']):
                if check_overlap(cur_motif_cluster_df.loc[i,'start'], cur_motif_cluster_df.loc[i,'stop'],
                                cur_motif_cluster_df.loc[j,'start'],cur_motif_cluster_df.loc[j,'stop'],overlap=overlap):
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
        cur_motif_cluster_df['representative'] = False
        for hit_cluster in overlapping_lists:
            # cur_motif_cluster_df.loc[hit_cluster[0],'representative'] = True
            cur_motif_cluster_df.loc[cur_motif_cluster_df.loc[hit_cluster,'p-value'].idxmin(),'representative'] = True
        pos_clustered_df = cur_motif_cluster_df[cur_motif_cluster_df['representative']==True]
        pos_clustered_df = pos_clustered_df.drop(columns=['representative'])
        pos_clustered_df = pos_clustered_df.reset_index(drop=True)

        # add cur_motif_cluster_df to new df
        if 'final_df' not in locals():
            final_df = pos_clustered_df
        else:
            final_df = pd.concat([final_df, pos_clustered_df])

    final_df = final_df.reset_index(drop=True)
    final_df['motif_alt_id'] = final_df['motif_alt_id'].str.upper()
    
    return final_df

def cluster_motifs_by_pos_v2(clustered_motif_df,overlap=3,by_score=False):

    seqs_with_hits = clustered_motif_df['sequence_name'].unique()

    for seq in seqs_with_hits:
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

### Plotting/visualization functions ###

# plot the start and stop positions of the motif hits in new_df on a line plot
### Make the y-axis log-scale ################################################################################################### TODO
def visualize_motif_hits(df,plot_names=False,hide_yaxis_label=False,figsize=(5,3)):

    if df.shape[0] == 0:
        print('No hits to visualize')
        return

    df = df.reset_index(drop=True)
    
    plt.figure(figsize=figsize)
    plt.yscale('log')
    plt.plot(df['start'], df['p-value'], 'o')
    plt.plot(df['stop'], df['p-value'], 'o')

    # draw line between start and stop
    for i in range(df.shape[0]):
        plt.plot([df.loc[i,'start'], df.loc[i,'stop']], [df.loc[i,'p-value'], df.loc[i,'p-value']], 'k-')

    if plot_names:
        # get range of y values
        # y_range = max(df['p-value']) - min(df['p-value'])
        y_offset = 1.25
        for i in range(df.shape[0]):
            plt.text(df.loc[i,'start'], df.loc[i,'p-value']*y_offset, df.loc[i,'motif_alt_id'])

    plt.legend(['start', 'stop'])
    plt.xlabel('Sequence position')
    if not hide_yaxis_label:
        plt.ylabel('p-value')
    plt.xlim([0,145])
    plt.title(f'Sequence: {df.loc[0,"sequence_name"]}')
    plt.show()

def plot_motif_repeats(seq_df,motif_df,motif_names,*args):
        # args should be 'h','k', and/or 'd' for HEPG2, K562, and H2K_deseq respectively

        seq_df = seq_df.copy()

        # if motif_names is not a list make it a list
        if type(motif_names) != list:
                motif_names = [motif_names]
        
        # create a new column in df that is the number of motif_names motifs in each sequence
        seq_df['n_cur_motif_repeats'] = 0
        for seq_idx in range(seq_df.shape[0]):
                if seq_idx in motif_df['sequence_name'].values:
                        seq_df.loc[seq_idx,f'n_cur_motif_repeats'] = motif_df[(motif_df['sequence_name']==seq_idx) & (motif_df['motif_alt_id'].isin(motif_names))].shape[0]

        # get unique values of n_cur_motif_repeats
        n_repeats = seq_df['n_cur_motif_repeats'].value_counts().index.sort_values()

        plt.figure(figsize=(5,3))
        if 'h' in args:
                plt.plot(n_repeats, [seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_HEPG2'].mean() for n in n_repeats], label='HEPG2',color='tab:orange')
                plt.errorbar(n_repeats, [seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_HEPG2'].mean() for n in n_repeats],
                        yerr=[seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_HEPG2'].sem() for n in n_repeats], fmt='o', color='k')

        if 'k' in args:
                plt.plot(n_repeats, [seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_K562'].mean() for n in n_repeats], label='K562',color='tab:blue')
                plt.errorbar(n_repeats, [seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_K562'].mean() for n in n_repeats],
                        yerr=[seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_K562'].sem() for n in n_repeats], fmt='o', color='k')

        if 'd' in args:
                plt.plot(n_repeats, [seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_H2K_deseq'].mean() for n in n_repeats], label='H2K',color='tab:green')
                plt.errorbar(n_repeats, [seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_H2K_deseq'].mean() for n in n_repeats],
                        yerr=[seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_H2K_deseq'].sem() for n in n_repeats], fmt='o', color='k')

        plt.legend()
        plt.xlabel('Number of motif repeats')
        plt.ylabel('Average log2FoldChange')
        plt.title(f'Effect of {motif_names} motif repeats')

        # return the number of seqs at each n_cur_motif_repeats, and the average log2FoldChange for each n_cur_motif_repeats
        return seq_df['n_cur_motif_repeats'].value_counts().sort_index(), [seq_df[seq_df['n_cur_motif_repeats']==n]['log2FoldChange_H2K_deseq'].mean() for n in n_repeats]