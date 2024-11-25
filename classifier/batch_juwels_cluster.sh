#!/bin/bash

# use project specified via jutil
#SBATCH --account=structuretofunction

# set the number of nodes
#SBATCH --nodes=1

# task per GPU
#SBATCH --ntasks-per-node=4

# big partition
#SBATCH --partition=batch

# set max wallclock time
#SBATCH --time=24:00:00

# enable email notifications
#SBATCH --mail-user=J.C.Knight@sussex.ac.uk
#SBATCH --mail-type=END,FAIL

# run the application
srun ./evaluate_args_juwels_cluster.sh
