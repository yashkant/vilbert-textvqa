import os
import json
import _pickle as cPickle
import logging
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
import multiprocessing as mp

data_root = "/nethome/ykant3/m4c-release/data/imdb/textvqa_0.5"

val_path = "imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_val.npy"
rev_val_path = "reverse_imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_val.npy"

val_path = os.path.join(data_root, val_path)
rev_val_path = os.path.join(data_root, rev_val_path)

os.path.exists(val_path)
os.path.exists(rev_val_path)

filter_dict = {
    "north": "south",
    "east": "west",
    "up": "down",
    "right": "left",
    "bottom": "top",
    "under": "over",
    "below": "above",
    "beside": None,
    "beneath": None
}

rev_dict = {}
for key, value in filter_dict.items():
    if value is not None:
        rev_dict[value] = key

filter_dict.update(rev_dict)


def word_cleaner(word):
    word = word.lower()
    word = word.replace(",", "").replace("?", "").replace("'s", " 's")
    return word.strip()


data = np.load(val_path, allow_pickle=True)
rev_data = [data[0]]


for instance in tqdm(data[1:]):
    question = instance["question"].split(" ")
    question = [word_cleaner(word) for word in question]
    spatial_words = []

    reversed_question = instance["question"]
    for word in question:
        if word in filter_dict.keys():
            spatial_words.append(word)
            if filter_dict[word] is not None:
                reversed_question = reversed_question.replace(word, filter_dict[word])

    if reversed_question != question:
        instance["original_question"] = instance["question"]
        instance["question"] = reversed_question
        instance["question_reversed"] = True
    else:
        instance["question_reversed"] = False

    instance["spatial_words"] = spatial_words

    if len(spatial_words) > 0:
        rev_data.append(instance)

import pdb
pdb.set_trace()
# 654/5000 contains spatial words that is 13.08% of validation set

np.save(rev_val_path, rev_data)

