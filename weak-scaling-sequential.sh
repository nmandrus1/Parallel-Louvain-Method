#!/bin/bash
#SBATCH --job-name=weak-scaling-seq-bfs
#SBATCH --output=weak_seq_bfs_%j.out
#SBATCH --error=weak_seq_bfs_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:6

# Adjust these as necessary for the test
EDGEFACTOR=16
SEED=123
PARALLEL=0
CUDA=0
CHECKPOINT_INTERVAL=0
OUTPUT=1

PROJECT="/gpfs/u/home/PCPE/PCPErsnl/barn/graph500/"
DATA_DIR="/gpfs/u/home/PCPE/PCPErsnl/barn/graph500/data"

module purge
source "/gpfs/u/home/PCPE/PCPErsnl/modules-to-load.sh"

# Loop over a range of scales, adjusting problem size
for SCALE in {4..16}
do
    mpirun -n 1 $PROJECT/src/graph500 $SCALE $EDGEFACTOR $SEED $PARALLEL $CUDA $CHECKPOINT_INTERVAL $OUTPUT > "$DATA_DIR/weak-seq-$SCALE.txt"
done

