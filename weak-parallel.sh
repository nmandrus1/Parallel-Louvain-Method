#!/bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=weak-scaling-par-bfs
#SBATCH --output=weak_par_bfs_%j.out
#SBATCH --error=weak_par_bfs_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:6


EDGEFACTOR=16
SEED=123
PARALLEL=1
CUDA=0
CHECKPOINT_INTERVAL=0
OUTPUT=1

PROJECT="/gpfs/u/home/PCPE/PCPErsnl/barn/graph500/"
DATA_DIR="/gpfs/u/home/PCPE/PCPErsnl/barn/graph500/data-nocuda"

module purge
source "/gpfs/u/home/PCPE/PCPErsnl/modules-to-load.sh"

# Loop to adjust CPU count based on scale
 for SCALE in {4..16}
 do
   if [ $SCALE -le 6 ]; then
       CPU_COUNT=4
   elif [ $SCALE -le 9 ]; then
       CPU_COUNT=9
   elif [ $SCALE -le 12 ]; then
       CPU_COUNT=16
   elif [ $SCALE -le 15 ]; then
       CPU_COUNT=25
   else
       CPU_COUNT=36
   fi

    # Construct the SLURM sbatch command to execute the graph500 with the right parameters
    echo "Running scale $SCALE with $CPU_COUNT CPUs"
    mpirun -n $CPU_COUNT $PROJECT/src/graph500 $SCALE $EDGEFACTOR $SEED $PARALLEL $CUDA $CHECKPOINT_INTERVAL $OUTPUT > "$DATA_DIR/weak-parallel-scale$SCALE-cpu$CPU_COUNT.txt"
done

