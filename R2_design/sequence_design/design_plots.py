import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import logomaker

# Copied from the SynthSeqs repo
DHS_COLORS = np.array([
    [195,195,195],
    [187,45,212],
    [5,193,217],
    [122,0,255],
    [254,129,2],
    [74,104,118],
    [255,229,0],
    [4,103,253],
    [7,175,0],
    [105,33,8],
    [185,70,29],
    [76,125,20],
    [0,149,136],
    [65,70,19],
    [255,0,0],
    [8,36,91],
]) / 255

# Nucleotide colors
# nt_color_dict = {
#     'A': 'darkgreen',
#     'C': 'blue',
#     'G': 'orange',
#     'T': 'red',
# }
nt_color_dict = {
    'A': (15/255, 148/255, 71/255),
    'C': (35/255, 63/255, 153/255),
    'G': (245/255, 179/255, 40/255),
    'T': (228/255, 38/255, 56/255),
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

    # Actually plot
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
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.015), loc='upper left', fontsize='medium')

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel('Position (nt)')
    ax.set_ylabel('Generated sequences')

    return fig

def plot_seq_logo(nt_height=None, seq_val=None, ax=None, title=None):
    """
    Plot a sequence logo
    
    TODO: add details
    """

    # If nt_height not provided directly, calculate from seq_val
    if nt_height is None:
        if type(seq_val)==np.ndarray:
            # Assume pwm
            pwm = seq_val
            entropy = np.zeros_like(pwm)
            entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
            entropy = np.sum(entropy, axis=1)
            conservation = 2 - entropy
            # Nucleotide height
            nt_height = np.tile(np.reshape(conservation, (-1, 1)), (1, 4))
            nt_height = pwm * nt_height
        elif type(seq_val)==str:
            # Assume string
            nt_to_onehot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
            nt_height = [nt_to_onehot[c] for c in seq_val]
            nt_height = np.array(nt_height)
        else:
            raise ValueError(f"type of seq_val {type(seq_val)} not recognized")

    nt_height_df = pd.DataFrame(
        nt_height,
        columns=['A', 'C', 'G', 'T'],
    )
    
    logo = logomaker.Logo(
        nt_height_df,
        # color_scheme='classic',
        color_scheme=nt_color_dict,
        ax=ax,
        font_name='Consolas',
    )
    logo.style_spines(visible=False)
    logo.style_spines(spines=['bottom'], visible=True, linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)

def plot_n_seqs(seq_vals, n_seqs=10):
    n_seqs = min(len(seq_vals), n_seqs)
    seq_len = len(seq_vals[0])

    fig, axes = plt.subplots(n_seqs, 1, figsize=(seq_len/10, 0.4*n_seqs))
    for seq_idx in range(n_seqs):
        plot_seq_logo(seq_val=seq_vals[seq_idx], ax=axes[seq_idx])

    return fig

def plot_bar_preds(signal_vals,cell_types=['HEPG2','K562','H2K'],ylabel="log10(accessibility)", figsize=None):

    signal_vals = np.array(signal_vals)
    signal_vals = np.hstack([signal_vals, np.expand_dims(signal_vals[:,0] - signal_vals[:,1], axis=1)])

    palette = ['tab:orange','tab:blue','tab:green']

    if figsize is None:
        figsize = ((len(cell_types)+1)*2, 4)
    fig, ax = plt.subplots(figsize=figsize)
    if len(np.squeeze(signal_vals).shape)==1:
        sns.barplot(
            x=np.arange(len(np.squeeze(signal_vals))),
            y=np.squeeze(signal_vals),
            palette=palette,
            ax=ax,
        )
    else:
        sns.barplot(
            signal_vals,
            errorbar='sd',
            palette=palette,
            ax=ax,
        )
    ax.set_xticklabels(cell_types, rotation=90, fontsize='small')
    ax.set_ylabel(ylabel)
    # ax.grid()

    # Custom legend
    legend_markers = [
        mpl.lines.Line2D([], [], color=c, marker='s', linestyle='None', markersize=7, label=name)
        for c, name in zip(palette,cell_types)
    ]
    ax.legend(handles=legend_markers, loc='upper left', bbox_to_anchor=(1.01, 1.025), fontsize='medium')

    return fig



def plot_bar_dhs_signal(signal_vals, cell_types, cell_type_metadata, ylabel="log10(accessibility)", figsize=None):

    signal_vals = np.array(signal_vals)
    
    # Determine colors
    dhs_component_info = cell_type_metadata.set_index('Biosample name').loc[cell_types, ['max. component', 'comp. no']]
    dhs_component_idx = dhs_component_info['comp. no'].to_numpy() - 1
    dhs_colors = [DHS_COLORS[i] for i in dhs_component_idx]

    # Determine legend info
    dhs_component_unique = dhs_component_info.drop_duplicates()
    dhs_component_unique_names = dhs_component_unique['max. component'].tolist()
    dhs_component_unique_colors = [DHS_COLORS[i] for i in (dhs_component_unique['comp. no'].to_numpy() - 1)]

    if figsize is None:
        figsize = (len(cell_types)/64*9, 4)
    fig, ax = plt.subplots(figsize=figsize)
    if len(np.squeeze(signal_vals).shape)==1:
        sns.barplot(
            x=np.arange(len(np.squeeze(signal_vals))),
            y=np.squeeze(signal_vals),
            palette=dhs_colors,
            ax=ax,
        )
    else:
        sns.barplot(
            signal_vals,
            errorbar='sd',
            palette=dhs_colors,
            ax=ax,
        )
    ax.set_xticklabels(cell_types, rotation=90, fontsize='small')
    ax.set_ylabel(ylabel)
    # ax.grid()

    # Custom legend
    legend_markers = [
        mpl.lines.Line2D([], [], color=c, marker='s', linestyle='None', markersize=7, label=name)
        for c, name in zip(dhs_component_unique_colors, dhs_component_unique_names)
    ]
    ax.legend(handles=legend_markers, loc='upper left', bbox_to_anchor=(1.01, 1.025), fontsize='medium')

    return fig

def plot_train_history(train_history):
    loss_components = ['loss', 'target_loss', 'pwm_loss', 'entropy_loss']
    loss_components = [c for c in loss_components if c in train_history]
    n_plots = len(loss_components)

    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 3))

    for plot_idx in range(n_plots):

        ax = axes[plot_idx]
        loss_component = loss_components[plot_idx]
        ax.plot(train_history[loss_component])
        ax.set_title(loss_component.replace('_', ' ').capitalize())
        ax.set_xlabel("Weight updates")

    return fig