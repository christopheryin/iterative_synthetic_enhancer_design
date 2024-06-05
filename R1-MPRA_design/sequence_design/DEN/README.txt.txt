To use DENs, a DEN model is first trained, then the trained DEN is used to generate sequences. A different DEN must be trained for different objective functions.

In this folder we provide example scripts for:

1. Training a DEN on a single model (train_DEN_*.ipynb)
2. Training a DEN on an ensemble model (train_ensemble_DEN_*.ipynb)
3. Generating sequences from a single model-trained DEN (design_single_DEN_h2k.ipynb)
4. Generating sequences form an ensemble model-trained DEN (design_ensemble_DEN_h2k.ipynb).

For the single DEN training scripts we provide scripts for training DENS to generate HepG2-specific sequences (_h2k suffix) or K562-specific sequences (_k2h suffix).

All scripts require tensorflow < 2 (the genesis package is incompatible with the latest versions of tensorflow, you will get bugs).