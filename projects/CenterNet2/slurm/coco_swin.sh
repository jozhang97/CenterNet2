#!/bin/bash

#SBATCH --job-name CenterNet2-Swin

### Logging
#SBATCH --output=/scratch/cluster/jozhang/logs/slurm/ct2_swin_%j.out # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/scratch/cluster/jozhang/logs/slurm/ct2_swin_%j.err  # Name of stderr output file (%j expands to jobId)
###SBATCH --mail-user=jozhang@cs.utexas.edu    # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

### Node info
#SBATCH --partition dgx                      # Queue name [NOT NEEDED FOR NOW]
#SBATCH --nodes=1                            # Always set to 1 when using the cluster
#SBATCH --time 72:00:00                      # Run time (hh:mm:ss)
#SBATCH --ntasks-per-node=1                  # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --gres=gpu:4                         # Number of gpus needed
#SBATCH --cpus-per-task=16                   # Number of cpus needed per task
#SBATCH --mem-per-cpu=12G                    # Memory requirements

export PYTHONNOUSERSITE=1
cd /u/jozhang/code/CenterNet2/projects/CenterNet2 || exit
source activate dgx
conda env list
python train_net.py --config-file configs/C2_SwinB_896b32_1x_adam.yaml \
    --num-gpus 4                                 \
    OUTPUT_DIR /scratch/cluster/jozhang/logs/ct2/CenterNet2-Swin \
    SOLVER.IMS_PER_BATCH 16  \
    SEED 42
