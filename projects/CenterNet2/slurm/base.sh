#!/bin/bash

#SBATCH --job-name Base-CenterNet2

### Logging
#SBATCH --output=/scratch/cluster/jozhang/logs/slurm/base_ct2_%j.out # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/scratch/cluster/jozhang/logs/slurm/base_ct2_%j.err  # Name of stderr output file (%j expands to jobId)
###SBATCH --mail-user=jozhang@cs.utexas.edu    # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

### Node info
#SBATCH --partition titans                   # Queue name [NOT NEEDED FOR NOW]
#SBATCH --nodes=1                            # Always set to 1 when using the cluster
#SBATCH --time 24:00:00                      # Run time (hh:mm:ss)
#SBATCH --ntasks-per-node=1                  # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --gres=gpu:4                         # Number of gpus needed
#SBATCH --cpus-per-task=8                    # Number of cpus needed per task
#SBATCH --mem-per-cpu=16G                    # Memory requirements
#SBATCH --exclude=titan-12                   # Bad GPU

cd /u/jozhang/code/CenterNet2/projects/CenterNet2 || exit
source activate titans
conda env list
python train_net.py --config-file configs/Base-CenterNet2.yaml \
    --num-gpus 4                                 \
    OUTPUT_DIR /scratch/cluster/jozhang/logs/ct2/Base-CenterNet2 \
    SEED 42
