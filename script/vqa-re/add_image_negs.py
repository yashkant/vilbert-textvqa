from copy import deepcopy

import numpy as np
from tqdm import tqdm
import json
import _pickle as cPickle

process_question_paths = [
    "../../data/re-vqa/data/revqa_train_proc.json",
    "../../data/re-vqa/data/revqa_val_proc.json"
]

process_question_image_negs_paths = [
    "../../data/re-vqa/data/revqa_train_proc_image_negs.json",
    "../../data/re-vqa/data/revqa_val_proc_image_negs.json"
]

answer_paths = [
    "../../datasets/VQA/cache/revqa_train_target.pkl",
    "../../datasets/VQA/cache/revqa_val_target.pkl"
]

answer_negs_paths = [
    "../../datasets/VQA/cache/revqa_train_target_image_negs.pkl",
    "../../datasets/VQA/cache/revqa_val_target_image_negs.pkl"
]

# original vqa split
val_path = "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json"
val_answers_path = "../../datasets/VQA/cache/val_target.pkl"

question_dict = {}
answer_dict = {}

for q in json.load(open(val_path))["questions"]:
    question_dict[q["question_id"]] = q

for a in cPickle.load(open(val_answers_path, "rb")):
    answer_dict[a["question_id"]] = a


for proc_que_file, proc_que_file_negs, ans_file, ans_file_negs in zip(
        process_question_paths,
        process_question_image_negs_paths,
        answer_paths,
        answer_negs_paths
    ):

    data = json.load(open(proc_que_file))
    new_questions = []
    new_answers = []
    new_questions_set = set()
    que_data  = json.load(open(proc_que_file))["questions"]
    ans_data = cPickle.load(open(ans_file, "rb"))

    for que, ans in tqdm(zip(que_data, ans_data), total=len(que_data), desc="Processing"):
        assert que["question_id"] == ans["question_id"]

        new_questions.append(que)
        new_answers.append(ans)
        new_questions_set.add(que["question_id"])

        for image_neg_id in que["same_image_questions"]:
            if image_neg_id in new_questions_set:
                continue

            neg_que = deepcopy(que)
            neg_que["valid_answers"] = None
            neg_que["question"] = question_dict[image_neg_id]["question"]
            neg_que["question_id"] = image_neg_id
            neg_que["same_image_questions"].remove(image_neg_id)
            neg_que["same_image_questions"].append(que["question_id"])


            if image_neg_id in neg_que["same_image_questions_neg"]:
                neg_que["same_image_questions_neg"].remove(image_neg_id)
                neg_que["same_image_questions_neg"].append(que["question_id"])

            neg_que["top_k_questions_neg"] = None
            neg_que["top_k_questions"] = None
            neg_que["rephrasing_ids"] = []
            new_questions.append(neg_que)

            neg_ans = answer_dict[image_neg_id]
            new_answers.append(neg_ans)

            new_questions_set.add(image_neg_id)

    assert len(new_answers) == len(new_questions) == len(set(new_questions_set))

    for q,a in zip(new_questions, new_answers):
        assert q["question_id"] == a["question_id"]

    # dump json and pkl files here.
    json.dump({
        "questions": new_questions,
        "data_type": "REVQA_with_image_negatives"
    }, open(proc_que_file_negs, "w"))

    cPickle.dump(new_answers, open(ans_file_negs, "wb"))

    print(f"Dumped Files: {proc_que_file_negs} and {ans_file_negs}")



























