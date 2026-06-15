#!/bin/bash

# Load Miniconda module
module load miniconda/24.9.2_environmentally

# Activate the Conda environment
conda activate my_env

# Run the Python script
python gen_dat_nk.py --N 1000 --K 4 --num_repeats 100

# Deactivate the environment after the script finishes
conda deactivate