import json
import os
import _pickle as cPickle
from copy import deepcopy

# Todo:
#   - Build json and pkl files
from tqdm import tqdm
from collections import Counter

val_path = "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json"
val_answers_path = "../../datasets/VQA/cache/val_target.pkl"

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

val_questions = val_data["questions"]
revqa_questions = revqa_data["questions"]

val_questions_ids = set([sample["question_id"] for sample in val_questions])
revqa_questions_ids = []


for sample in revqa_questions:
    if "rephrasing_of" in sample:
        revqa_questions_ids.append(sample["rephrasing_of"])

freq_counter = Counter(revqa_questions_ids)
# Few of them have 4 re-phrasings.
print("Most Common: ", freq_counter.most_common(10))

# # There are colliding question-ids! :/
# for _q in revqa_questions:
#     if "rephrasing_of" in _q:
#         if _q["question_id"] in val_questions_ids:
#             import pdb
#             pdb.set_trace()

overlap_question_ids = list(set(val_questions_ids.intersection(set(revqa_questions_ids))))
overlap_question_ids_splits = [set(overlap_question_ids[:-5000]), set(overlap_question_ids[-5000:])]
questions_splits = [[], []]

for _q in tqdm(revqa_questions):
    source_id = _q["rephrasing_of"] if "rephrasing_of" in _q else _q["question_id"]
    if source_id in overlap_question_ids_splits[0]:
        assert source_id not in overlap_question_ids_splits[1]
        questions_splits[0].append(_q)

    else:
        assert source_id in overlap_question_ids_splits[1]
        questions_splits[1].append(_q)

assert len(questions_splits[0]) + len(questions_splits[1]) == len(revqa_questions)
# questions = sorted(questions, key=lambda x: x["question_id"])

for idx in range(len(questions_splits)):
    questions_splits[idx] = sorted(questions_splits[idx], key=lambda x: x["question_id"])

answers_val_dict = {}
for answer in answers_val:
    answers_val_dict[answer["question_id"]] = answer

answer_splits = [[], []]

for idx in range(len(questions_splits)):
    for _q in questions_splits[idx]:
        source_id = _q["rephrasing_of"] if "rephrasing_of" in _q else _q["question_id"]
        answer = deepcopy(answers_val_dict[source_id])
        answer["question_id"] = _q["question_id"]
        answer_splits[idx].append(answer)

rephrasings_of = [[], []]
idx = -1
for (q_split, a_split) in zip(questions_splits, answer_splits):
    idx += 1
    for (q, a) in zip(q_split, a_split):
        assert q["question_id"] == a["question_id"]
        if "rephrasing_of" in q:
            rephrasings_of[idx].append(q["rephrasing_of"])

# Source-ids intersection should be None
x = set(rephrasings_of[0]).intersection(rephrasings_of[1])
assert len(x) == 0

# assert no-collisions
split1_ids = [x["question_id"] for x in questions_splits[0]]
split2_ids = [x["question_id"] for x in questions_splits[1]]
assert len(split1_ids) == len(set(split1_ids))
assert len(split2_ids) == len(set(split2_ids))

for ans_split, que_split, ans_path, que_path, split in zip(answer_splits, questions_splits, answer_paths, json_paths,
                                                           ["train", "val"]):
    revqa_data = {
        "questions": que_split,
        "data_subtype": f"val2014_rephrasings_{split}",
    }
    json.dump(revqa_data, open(que_path, "w"))
    cPickle.dump(ans_split, open(ans_path, "wb"))
    print(f"Dumping: {que_path}; {ans_path}")
