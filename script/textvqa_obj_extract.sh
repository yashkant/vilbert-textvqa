#!/bin/bash

#SBATCH -p  long
#SBATCH --gres=gpu:1
#SBATCH -J feature-extraction-textvqa
#SBATCH -o feature_obj_textvqa_test.txt
#SBATCH -w hal

hostname
echo $CUDA_AVAILABLE_DEVICES

srun python extract_features.py
