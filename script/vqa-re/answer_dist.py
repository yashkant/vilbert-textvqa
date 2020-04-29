from collections import defaultdict, Counter
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import json
import _pickle as cPickle
import os
import matplotlib.pyplot as plt


process_question_paths = [
    "../../data/re-vqa/data/revqa_train_proc.json",
    "../../data/re-vqa/data/revqa_val_proc.json"
]

answer_paths = [
    "../../datasets/VQA/cache/revqa_train_target.pkl",
    "../../datasets/VQA/cache/revqa_val_target.pkl",
]

ans2label_path = os.path.join("../../datasets/VQA/", "cache", "trainval_ans2label.pkl")
label2ans_path = os.path.join("../../datasets/VQA/", "cache", "trainval_label2ans.pkl")
ans2label = cPickle.load(open(ans2label_path, "rb"))
label2ans = cPickle.load(open(label2ans_path, "rb"))

counters = []

for que_path, ans_path, plot_name in zip(process_question_paths, answer_paths, ["train_ans.pdf", "val_ans.pdf"]):
    cnt = Counter()
    answers = cPickle.load(open(ans_path, "rb"))

    for ans in answers:
        for label in ans["labels"]:
            cnt[label] += 1

    answers, frequency = list(zip(*cnt.most_common()))
    plt.clf()
    plt.bar(answers[2:], frequency[2:])
    plt.savefig(plot_name)
