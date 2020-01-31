#!/bin/bash

#SBATCH -p  long
#SBATCH --gres=gpu:2
#SBATCH -J vilbert-textvqa
#SBATCH -o jlogs/vilbert-multi.txt
#SBATCH -x neo,kipp,calculon,ripl-s1,ash,ava,siri,johnny5,irona,cortana
#SBATCH -w hal

host_name=$(srun hostname)
echo $host_name

## NOTE: make sure you use the right node due to high I/O operations

srun \
python train_tasks.py \
--task_file vilbert_tasks.yml \
--bert_model bert-base-uncased \
--from_pretrained data/multitask_model/pytorch_model_14.bin \
--config_file config/bert_base_6layer_6conect_textvqa.json \
--tasks 19  --lr_scheduler 'warmup_linear' \
--train_iter_gap 4 --save_name finetune_from_multi_task_model \
--optim "AdamW" --tag "adamw-2-2-multi-14" --model_type "22"
