#!/bin/bash
# Read mode and pop
mode=$1
shift

for s in 1234 2345 3456 4567 5678
do
    # SHD
    python classifier.py --mode $mode --num-epochs 50 --dataset shd --augmentation shift-blend --num-hidden 256 --reg-lambda 2.5e-10 --dt 1 --num-timesteps 1000 --seed $s $@
    python classifier.py --mode $mode --num-epochs 50 --dataset shd --augmentation shift-blend --num-hidden 512 --reg-lambda 5e-10 --dt 1 --num-timesteps 1000 --seed $s $@
    python classifier.py --mode $mode --num-epochs 50 --dataset shd --augmentation shift-blend --num-hidden 1024 --reg-lambda 5e-11 --dt 1 --num-timesteps 1000 --seed $s $@
    
    # SSC
    python classifier.py --mode $mode --num-epochs 100 --dataset ssc --augmentation shift --num-hidden 256 --reg-lambda 2.5e-9 --dt 1 --num-timesteps 1000 --seed $s $@
    python classifier.py --mode $mode --num-epochs 100 --dataset ssc --augmentation shift --num-hidden 512 --reg-lambda 5e-10 --dt 1 --num-timesteps 1000 --seed $s $@
    python classifier.py --mode $mode --num-epochs 100 --dataset ssc --augmentation shift --num-hidden 1024 --reg-lambda 2.5e-10 --dt 1 --num-timesteps 1000 --seed $s $@
done



