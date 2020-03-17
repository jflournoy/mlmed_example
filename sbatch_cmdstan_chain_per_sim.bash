#!/bin/bash
#
#SBATCH --job-name=testsim
#SBATCH -o %x_%A_%a.log
#SBATCH --time=9-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=ncf
#SBATCH --mem=5000
#SBATCH --mail-type=END

module load gcc/8.2.0-fasrc01  
export STAN_NUM_THREADS=${SLURM_CPUS_PER_TASK}
id=$SLURM_ARRAY_TASK_ID

CMDSTANDIR=/users/jflournoy/otherhome/code/cmdstan/mlmed
pushd $CMDSTANDIR

nicei=$(printf "%02d" $id)
echo $nicei

srun -c 1 time ./mlmed_example \
  sample \
    adapt delta=.99 \
    algorithm=hmc engine=nuts max_depth=20 \
    num_samples=1000 num_warmup=1000 \
  random seed=6868 id="${id}" \
  output file=../../mlmed_example/fit"${nicei}".csv \
  data file=../../mlmed_example/simdata"${nicei}".R

