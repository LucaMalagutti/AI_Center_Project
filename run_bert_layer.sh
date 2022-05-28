n_seeds=$1
layer=$2
for (( c=1; c<= n_seeds; c++ ))
do
   cmd="python train.py --bert_stem_weighted=0 --random_seed=${c} --bert_layer=${layer} --dataset=wn18rr --model=tucker --init=bert --embdim=256 --epochs=500 --wandb_group=std_runs_bert_256_layer_${layer}"
   bsub  -n 16 -W 11:59 -R "rusage[mem=5000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" ${cmd}
   sleep 1
done
