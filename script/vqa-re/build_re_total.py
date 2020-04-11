import json
import os
import _pickle as cPickle
from copy import deepcopy

val_path = "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json"
val_answers_path = "../../datasets/VQA/cache/val_target.pkl"
revqa_path = "../../data/re-vqa/data/v2_OpenEnded_mscoco_valrep2014_humans_og_questions_organized.json"

revqa_total_path = "../../data/re-vqa/data/revqa_total.json"
revqa_total_answers_path = "../../datasets/VQA/cache/revqa_total_target.pkl"


val_data = json.load(open(val_path))
revqa_data = json.load(open(revqa_path))

val_questions = val_data["questions"]
revqa_questions = revqa_data["questions"]

val_questions_ids = set([sample["question_id"] for sample in val_questions])
revqa_questions_ids = set([sample["question_id"] for sample in revqa_questions])
overlap_question_ids = val_questions_ids.intersection(revqa_questions_ids)

overlap_questions = []
for question in revqa_questions:
    if question["question_id"] in overlap_question_ids or question.get("rephrasing_of", -1) in overlap_question_ids:
        overlap_questions.append(question)

assert len(overlap_questions) == len(revqa_questions)

# dump new data
overlap_questions = sorted(overlap_questions, key=lambda x: x["question_id"])
revqa_data["questions"] = overlap_questions
revqa_data['data_subtype'] = "val2014_rephrasings_total"
json.dump(revqa_data, open(revqa_total_path, "w"))
print(f"Dumping: {revqa_total_path}")

answer_path_val = os.path.join(val_answers_path)
answers_val = cPickle.load(open(answer_path_val, "rb"))

answers_val_dict = {}
for answer in answers_val:
    answers_val_dict[answer["question_id"]] = answer

revqa_total_answers = []
for question in revqa_data["questions"]:
    question_id = question["question_id"]

    if "rephrasing_of" in question:
        question_id = question["rephrasing_of"]

    answer = deepcopy(answers_val_dict[question_id])
    answer["question_id"] = question["question_id"]
    revqa_total_answers.append(answer)

for (q, a) in zip(overlap_questions, revqa_total_answers):
    assert q["question_id"] == a["question_id"]

cPickle.dump(revqa_total_answers, open(revqa_total_answers_path, "wb"))
print(f"Dumping: {revqa_total_answers_path}")