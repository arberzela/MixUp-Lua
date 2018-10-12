#!/bin/bash
#
# szbmit to the right queue
#SBATCH -p meta_gpu-ti
#SBATCH --gres gpu:1
#SBATCH -a 1-5
#
# the execution will use the current directory for execution (important for relative paths)
#SBATCH -D .
#
# redirect the output/error to some files
#SBATCH -e .%A_%a.e
#SBATCH -o .%A_%a.o
#
th main.lua -dataset cifar10 -nGPU 1 -batchSize 64 -depth 26 -shareGradInput false -optnet true -nEpochs 1800 -netType shakeshake -lrShape cosine -baseWidth 96 -LR 0.1 -forwardShake true -backwardShake true -shakeImage true -Te 120 -Tmult 2 -widenFactor 1 -irun $SLURM_ARRAY_TASK_ID -cutout_half_size 8 -alpha 0.2
