#!/bin/bash

#SBATCH -p  long
#SBATCH --gres=gpu:1
#SBATCH -J feature-extraction-textvqa
#SBATCH -o feature_ocr_textvqa_val.txt
#SBATCH -w hal

hostname
echo $CUDA_AVAILABLE_DEVICES

srun python extract_features_ocr.py