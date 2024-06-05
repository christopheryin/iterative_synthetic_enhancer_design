Example script for generating sequences with the seqprop package (requires downloading https://github.com/johli/seqprop and installing via setup.py prior to running).

Note that the seqprop package available at this github link requires tensorflow < 2, otherwise you will get runtime errors.

Unfortunately I did not write this script very cleanly, and hard-coded several important values buried in the code:

- fitness_target (value used to clip the maximum predicted |log2FC_H2K|)
- fitness_loss (you have to manually change the subtraction order to design for HepG2-specific or K562-specific sequences).