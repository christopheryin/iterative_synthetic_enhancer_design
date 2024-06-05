process_fimo.tsv should be run from the command line. It takes in the output .tsv from the FIMO program of MEMESuite (https://meme-suite.org/meme/doc/fimo.html?man_type=web; you may need to remove the final 3 rows of annotation text from this .tsv before running otherwise errors will occur), as well as a dataframe containing at least a sequence_name column that corresponds exactly with the sequence name values used to generate the fimo tsv.

The script will output 3 files -

- id_clustered (collapses together all overlapping motifs with the same alt_id into the motif with the lowest p-value)

- pos_clustered (collapses together all overlapping motifs into the motif with the lowest p_value)

- seq_df_plus_cluster_counts (annotates provided sequence dataframe by adding columns for each motif cluster with entries corresponding to the number of that motif cluster found in a given sequence (row))