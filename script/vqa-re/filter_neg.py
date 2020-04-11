from collections import defaultdict
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import json
import _pickle as cPickle
import os

process_question_paths = [
    "../../data/re-vqa/data/revqa_train_proc.json",
    "../../data/re-vqa/data/revqa_val_proc.json"
]

answer_paths = [
    "../../datasets/VQA/cache/revqa_train_target.pkl",
    "../../datasets/VQA/cache/revqa_val_target.pkl",
    "../../datasets/VQA/cache/val_target.pkl"
]

ans2label_path = os.path.join("../../datasets/VQA/", "cache", "trainval_ans2label.pkl")
label2ans_path = os.path.join("../../datasets/VQA/", "cache", "trainval_label2ans.pkl")
ans2label = cPickle.load(open(ans2label_path, "rb"))
label2ans = cPickle.load(open(label2ans_path, "rb"))

answer_dict = {}
for path in tqdm(answer_paths):
    answers = cPickle.load(open(path, "rb"))

    for ans in answers:
        answer_dict[ans["question_id"]] = ans["labels"]

question_dict = {}
questions_rephrasings = defaultdict(list)
for que_path in process_question_paths:
    data = json.load(open(que_path, "r"))

    for sample in data["questions"]:
        question_dict[sample["question_id"]] = sample["question"]
        source_id = sample["rephrasing_of"] if "rephrasing_of" in sample else sample["question_id"]
        questions_rephrasings[source_id].append(sample["question_id"])


def add_rephrasing_ids(sample):
    source_id = sample["rephrasing_of"] if "rephrasing_of" in sample else sample["question_id"]
    rephrasing_ids = deepcopy(questions_rephrasings[source_id])
    rephrasing_ids.remove(sample["question_id"])
    sample["rephrasing_ids"] = rephrasing_ids


def filter_negatives(sample):
    # filter same-image questions
    same_image_ids = sample["same_image_questions"]
    fil_same_image_ids = []
    ref_answers = answer_dict[sample["question_id"]]
    for qid in same_image_ids:
        cand_answers = answer_dict[qid]
        if len(set(ref_answers).intersection(set(cand_answers))) == 0:
            fil_same_image_ids.append(qid)
    sample["same_image_questions_neg"] = fil_same_image_ids

    # filter top-k questions
    top_k_questions = sample["top_k_questions"]
    fil_top_k_questions = []
    for qid in top_k_questions:
        cand_answers = answer_dict[qid]
        if len(set(ref_answers).intersection(set(cand_answers))) == 0:
            fil_top_k_questions.append(qid)
    sample["top_k_questions_neg"] = fil_top_k_questions

    # print(f"Ref: {sample['question']}, Ans: {[label2ans[x] for x in ref_answers]}")
    # for qid in fil_top_k_questions[:10]:
    #     print(f"Neg Cand: {question_dict[qid]}, Ans: {[label2ans[x] for x in answer_dict[qid]]}")
    #
    # import pdb
    # pdb.set_trace()

for que_path in process_question_paths:
    data = json.load(open(que_path, "r"))

    for sample in tqdm(data["questions"]):
        filter_negatives(sample)
        add_rephrasing_ids(sample)

    json.dump(data, open(que_path, "w"))
    print(f"Dumping File: {que_path}")


