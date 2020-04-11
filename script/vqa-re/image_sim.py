import json
import os
import _pickle as cPickle
from copy import deepcopy

from tqdm import tqdm
from collections import Counter, defaultdict

val_path = "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json"
train_path = "../../datasets/VQA/v2_OpenEnded_mscoco_train2014_questions.json"
val_answers_path = "../../datasets/VQA/cache/val_target.pkl"
train_answers_path = "../../datasets/VQA/cache/train_target.pkl"

org_revqa_path = "../../data/re-vqa/data/v2_OpenEnded_mscoco_valrep2014_humans_og_questions_organized.json"

json_paths = [
    "../../data/re-vqa/data/revqa_train.json",
    "../../data/re-vqa/data/revqa_val.json"
]

answer_paths = [
    "../../datasets/VQA/cache/revqa_train_target.pkl",
    "../../datasets/VQA/cache/revqa_val_target.pkl"
]


val_data = json.load(open(val_path))
revqa_data = json.load(open(org_revqa_path))
answers_val = cPickle.load(open(val_answers_path, "rb"))

revqa_image_dict = defaultdict(list)
val_image_dict = defaultdict(list)

for sample in revqa_data["questions"]:
    revqa_image_dict[sample["image_id"]].append(sample["question_id"])

for sample in val_data["questions"]:
    val_image_dict[sample["image_id"]].append(sample["question_id"])

image_intersect_ids = set(val_image_dict.keys()).intersection(set(revqa_image_dict.keys()))
len_other_questions = []

for sample in tqdm(revqa_data["questions"]):
    image_id = sample["image_id"]
    source_id = sample["rephrasing_of"] if "rephrasing_of" in sample else sample["question_id"]
    other_questions = deepcopy(val_image_dict[image_id])
    if source_id in other_questions:
        other_questions.remove(source_id)
    sample["same_image_questions"] = other_questions
    len_other_questions.append(len(other_questions))

json.dump(revqa_data, open(org_revqa_path, "w"))
print(f"Saved File: {org_revqa_path}")
