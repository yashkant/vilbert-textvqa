#!/bin/bash

#SBATCH -p  long
#SBATCH --gres=gpu:1
#SBATCH -J build-splits-job
#SBATCH -o logs/batch-jobs.txt
#SBATCH -x gideon,irona,cortana,calculon,neo,vincent,johnny5,vicki,ava,kipp,ripl-s1,breq,ephemeral-3,siri,bmo,t1000,alexa,droid,rosie,asimo,pops

host_name=$(srun hostname)
echo $host_name, seq-id:$1
srun python build_splits.py --seq_id $1 > logs/build-splits-$1.txt
