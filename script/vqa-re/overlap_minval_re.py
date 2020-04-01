import json
import os
import _pickle as cPickle
from copy import deepcopy

val_path = "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json"
revqa_path = "../../data/re-vqa/data/v2_OpenEnded_mscoco_valrep2014_humans_og_questions.json"
revqa_minval_path = "../../data/re-vqa/data/revqa_minval_intersect.json"
val_answers_path = "../../datasets/VQA/cache/val_target.pkl"
overlap_val_answers_path = "../../datasets/VQA/cache/revqa_minval_intersect_target.pkl"


val_data = json.load(open(val_path))
revqa_data = json.load(open(revqa_path))

minval_questions = val_data["questions"][-3000:]
revqa_questions = revqa_data["questions"]

minval_questions_ids = set([sample["question_id"] for sample in minval_questions])
revqa_questions_ids = set([sample["question_id"] for sample in revqa_questions])

overlap_question_ids = minval_questions_ids.intersection(revqa_questions_ids)

overlap_questions = []
for question in revqa_questions:
    if question["question_id"] in overlap_question_ids or question.get("rephrasing_of", -1) in overlap_question_ids:
        overlap_questions.append(question)

# dump new data
overlap_questions = sorted(overlap_questions, key=lambda x: x["question_id"])
revqa_data["questions"] = overlap_questions
revqa_data['data_subtype'] = "val2014_rephrasings_minval_overlap"
json.dump(revqa_data, open(revqa_minval_path, "w"))


answer_path_val = os.path.join(val_answers_path)
answers_val = cPickle.load(open(answer_path_val, "rb"))

answers_val_dict = {}
for answer in answers_val:
    answers_val_dict[answer["question_id"]] = answer

overlap_answers_val = []
for question in revqa_data["questions"]:
    question_id = question["question_id"]

    if "rephrasing_of" in question:
        question_id = question["rephrasing_of"]

    answer = deepcopy(answers_val_dict[question_id])
    answer["question_id"] = question["question_id"]
    overlap_answers_val.append(answer)

for (q, a) in zip(overlap_questions, overlap_answers_val):
    assert q["question_id"] == a["question_id"]

cPickle.dump(overlap_answers_val, open(overlap_val_answers_path, "wb"))
