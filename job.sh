#!/bin/bash

#SBATCH -p  long
#SBATCH --gres=gpu:1
#SBATCH -J vilbert-textvqa
#SBATCH -o jlogs/adamw-lr-1e4-warm-10-bs-96-32-model-22ss.txt
#SBATCH -x neo,kipp,calculon,ripl-s1,ash,ava,siri,johnny5,irona,cortana

host_name=$(srun hostname)
echo $host_name

## NOTE: make sure you use the right node due to high I/O operations

srun \
python train_tasks.py \
--task_file sweeps/adamw-lr-1e4-warm-10\|bs-96-32\|model-22ss.yml \
--from_scratch \
--config_file config/small_bert_base_3layer_2conect_textvqa.json \
--tasks 19 \
--train_iter_gap 4 --save_name finetune_from_multi_task_model \
--tag "adamw-lr-1e4-warm-10-bs-96-32-model-22ss"