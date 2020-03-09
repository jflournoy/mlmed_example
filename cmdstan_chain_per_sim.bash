#!/bin/bash

CMDSTANDIR=~/code_new/cmdstan/mlmed2
pushd $CMDSTANDIR

for i in {1..8}
do
  nicei=$(printf "%02d" $i)
  echo $nicei
  
  time ./mlmed_example \
    sample \
      algorithm=hmc engine=nuts max_depth=15 \
      num_samples=1000 num_warmup=1000 \
    random seed=6868 id="${i}" \
    output file=../../mlmed_example/fit"${nicei}".csv \
    data file=../../mlmed_example/simdata"${nicei}".R > fit"${nicei}".log &
done

popd