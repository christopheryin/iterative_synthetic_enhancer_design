# Iterative deep learning-design of human enhancers exploits condensed sequence grammar to achieve cell type-specificity

This repository contains data analysis and sequence design code from "Iterative deep learning-design of human enhancers exploits condensed sequence grammar to achieve cell type-specificity" (paper link forthcoming).

## Contents
 
### R1-MPRA_design

- model_training - example scripts for training models used in R1-MPRA design

- sequence_design - example scripts used to generate R1-MPRA enhancer designs. Requires installation of: [seqprop](https://github.com/johli/seqprop), [genesis](https://github.com/johli/genesis).

### R2_design

- model_training - example scripts for training models used in R2 design

- sequence_design - example scripts used to generate R2 enhancer designs.

### analysis

- fimo_processing - code for processing FIMO output .tsv files and performing custom position- and identity-based clustering

- model_interpretation - example script for computing SHAP values with the models used to design enhancer libraries (implements [shap](https://github.com/shap/shap))

- paper_figures - jupyter notebooks and associated utility scripts for generating all the main and supplementary figures in the paper, includes zipped processed data



###
