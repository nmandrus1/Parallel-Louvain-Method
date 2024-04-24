#!/bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=strong-scaling-par-bfs
#SBATCH --output=strong_par_bfs_%j.out
#SBATCH --error=strong_par_bfs_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:6

EDGEFACTOR=16
SEED=123
PARALLEL=1
CUDA=1
CHECKPOINT_INTERVAL=1
OUTPUT=1

PROJECT="/gpfs/u/home/PCPE/PCPErsnl/barn/graph500/"
DATA_DIR="/gpfs/u/home/PCPE/PCPErsnl/barn/graph500/data-cuda-checkpoints"

module purge

source "/gpfs/u/home/PCPE/PCPErsnl/modules-to-load.sh"

# Test with increasing square values of CPU count
#for CPU_COUNT in 9 16 25 36
#do
#   # Loop to adjust SCALE count based on CPU_COUNT
#   if [ $CPU_COUNT -le 9 ]; then
#       SCALE=18
#   elif [ $CPU_COUNT -le 16 ]; then
#       SCALE=17
#   elif [ $CPU_COUNT -le 25 ]; then
#       SCALE=17
#   else
#       SCALE=16
#   fi

for CPU_COUNT in 16 25 36
do
   # Loop to adjust SCALE count based on CPU_COUNT
   if [ $CPU_COUNT -le 16 ]; then
       SCALE=17
   elif [ $CPU_COUNT -le 25 ]; then
       SCALE=17
   else
       SCALE=16
   fi



    mpirun -n $CPU_COUNT $PROJECT/src/graph500 $SCALE $EDGEFACTOR $SEED $PARALLEL $CUDA $CHECKPOINT_INTERVAL $OUTPUT > "$DATA_DIR/strong-parallel-$CPU_COUNT.txt"
done

