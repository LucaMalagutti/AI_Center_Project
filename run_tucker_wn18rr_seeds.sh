#!/bin/bash
# 1st argument: number of seeds
# 2nd argument: embedding mode
# e.g. : ./run_tucker_wn188rr_seeds 10 glove

n_seeds=$1
emb=$2
for (( c=1; c<= n_seeds; c++ ))
do  
   cmd="python train.py --dataset=wn18rr --model=tucker --init=${emb} --embdim=200 --epochs=250 --wandb_group=std_runs_${emb} --random_seed=${c}" 
   bsub  -n 16 -W 11:59 -R "rusage[mem=5000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" ${cmd}
   sleep 1
done
