Originally R0-MPRA was referred to as D1, R1-MPRA as D2, R1-DHS as DW, and the majority of code follows this convention.

train_M0_models.py - trains the 9 "M0" models on the DESeq2-processed crossfolds of the R0-MPRA dataset, using newly optimized hyperparameters. These models allow for up to 200bp input sequence length to account for eventual training on R1-DHS data.

train_M0_1_finetuned_models.py - finetunes each of the 9 "M0" models on R1 datasets, to produce the "M0+1" models.

train_M1_models.py - trains the 9 "M1" models directly on R1 libraries without pretraining