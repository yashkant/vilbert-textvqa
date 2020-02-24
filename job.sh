#!/bin/bash

#SBATCH -p  long
#SBATCH --gres=gpu:2
#SBATCH -J vilbert-textvqa
#SBATCH -o jlogs/m4c-spatial-mask-1-2-8-layers-4.txt
#SBATCH -x vicki,neo,kipp,calculon,ripl-s1,ash,ava,siri,johnny5,irona,cortana,ephemeral-3,siri,rosie,smith,bmo,pops


host_name=$(srun hostname)
echo $host_name

## NOTE: make sure you use the right node due to high I/O operations

srun \
python train_tasks.py \
--task_file sweeps/m4c-spatial-mask-1-2-8-layers-4.yml \
--from_scratch \
--config_file config/spatial_m4c_mmt_textvqa.json \
--tasks 19 \
--train_iter_gap 4 --save_name finetune_from_multi_task_model \
--tag "m4c-spatial-mask-1-2-7-layers-4"


python evaluate_textvqa.py \
--task_file sweeps/m4c-spatial-mask-1-2-layers-4.yml \
--config_file config/spatial_m4c_mmt_textvqa.json \
--vocab_file /srv/share3/hagrawal9/project/m4c/data/m4c_vocabs/textvqa/fixed_answer_vocab_textvqa_5k.txt \
--batch_size 96 \
--model_ckpt save/TextVQA_spatial_m4c_mmt_textvqa-finetune_from_multi_task_model-local-spatial-4layers-mask-1-2/pytorch_ckpt_latest.tar \
--beam_size 1 \
--save_file results_bs=1.df