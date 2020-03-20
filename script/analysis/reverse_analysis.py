import os
import json
import _pickle as cPickle
from collections import defaultdict

import logging
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
import multiprocessing as mp

def find_qid(list, qid):
    for x in list:
        if x["question_id"] == qid:
            return x
    return None


baseline_path = "/nethome/ykant3/tmp/m4c-release/analysis-reverse/baseline-layers6-beam5.json"
rev_baseline_path = "/nethome/ykant3/tmp/m4c-release/analysis-reverse/baseline-reverse-beam5.json"

best_model_path = "/nethome/ykant3/tmp/m4c-release/analysis-reverse/share3.json"
rev_best_model_path = "/nethome/ykant3/tmp/m4c-release/analysis-reverse/reverse_share2.json"

data_root = "/nethome/ykant3/m4c-release/data/imdb/textvqa_0.5"
rev_val_path = "reverse_imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_val.npy"
rev_val_path = os.path.join(data_root, rev_val_path)
rev_data = np.load(rev_val_path, allow_pickle=True)

image_id_instance_dict = defaultdict(list)
que_id_instance_dict = {}

baseline_data = json.load(open(baseline_path))
rev_baseline_data = json.load(open(rev_baseline_path))

best_data = json.load(open(best_model_path))
rev_best_data = json.load(open(rev_best_model_path))

for instance in rev_data:
    if "image_id" in instance:
        question_id = instance["question_id"]
        image_id = instance["image_id"]
        rev_instance_que = instance["question"]
        instance_que = instance["original_question"]
        instance = {}
        instance["baseline_ans"] = find_qid(baseline_data, question_id)
        instance["baseline_ans_rev"] = find_qid(rev_baseline_data, question_id)
        instance["best_ans"] = find_qid(best_data, question_id)
        instance["best_ans_rev"] = find_qid(rev_best_data, question_id)
        instance["rev_question"] = rev_instance_que
        instance["original_question"] = instance_que

        if instance["best_ans_rev"] != instance["baseline_ans_rev"]:
            image_id_instance_dict[image_id].append(instance)

import pdb
pdb.set_trace()

np.save("reverse_analysis.npy",image_id_instance_dict)