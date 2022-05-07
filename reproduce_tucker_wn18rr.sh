#!/bin/bash
embds="w2v baseline"
dims="200 300"
for emb in ${embds}; do
    for dim in ${dims}; do
    	cmd="python train.py --dataset=wn18rr --model=tucker --init=${emb} --embdim=${dim}"
		bsub  -n 16 -W 11:59 -R "rusage[mem=5000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" ${cmd}
		sleep 1
	done
done
embds="glove" 
dims="200"
for emb in ${embds}; do
    for dim in ${dims}; do
    	cmd="python train.py --dataset=wn18rr --model=tucker --init=${emb} --embdim=${dim}"
		bsub  -n 16 -W 11:59 -R "rusage[mem=5000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" ${cmd}
		sleep 1
    done
done

embds="bert" 
dims="256"
for emb in ${embds}; do
    for dim in ${dims}; do
    	cmd="python train.py --dataset=wn18rr --model=tucker --init=${emb} --embdim=${dim}"
		bsub  -n 16 -W 11:59 -R "rusage[mem=5000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" ${cmd}
		sleep 1
    done
done
