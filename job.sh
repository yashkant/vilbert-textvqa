#!/bin/bash

#SBATCH -p  long
#SBATCH --gres=gpu:2
#SBATCH -J vilbert-textvqa
#SBATCH -o jlogs/m4c-spatial-mask-all-implicit-6-layers-4.txt
#SBATCH -x neo,kipp,calculon,ripl-s1,ash,ava,siri,johnny5,irona,cortanac,ephemeral-3,siri,rosie,smith,bmo


host_name=$(srun hostname)
echo $host_name

## NOTE: make sure you use the right node due to high I/O operations

srun \
python train_tasks.py \
--task_file sweeps/m4c-spatial-mask-all-implicit-6-layers-4.yml  \
--from_scratch \
--config_file config/spatial_m4c_mmt_textvqa.json \
--tasks 19 \
--train_iter_gap 4 --save_name finetune_from_multi_task_model \
--tag "m4c-spatial-mask-all-implicit-6-layers-4"
