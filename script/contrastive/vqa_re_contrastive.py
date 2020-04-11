import json
import os
import _pickle as cPickle
from collections import defaultdict, Counter
from copy import deepcopy

# Todo:
#   - Build json and pkl files

val_path = "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json"
val_answers_path = "../../datasets/VQA/cache/val_target.pkl"


revqa_path = "../../data/re-vqa/data/v2_OpenEnded_mscoco_valrep2014_humans_og_questions.json"

question_paths = [
    "../../data/re-vqa/data/revqa_train.json",
    "../../data/re-vqa/data/revqa_val.json"
]

answer_paths = [
    "../../datasets/VQA/cache/revqa_train_target.pkl",
    "../../datasets/VQA/cache/revqa_val_target.pkl"
]

contrast_info_path = "../../data/re-vqa/data/contrast_info.json"

val_data = json.load(open(val_path))
revqa_data = json.load(open(revqa_path))
answers_val = cPickle.load(open(val_answers_path, "rb"))

val_questions = val_data["questions"]
revqa_questions = revqa_data["questions"]

val_questions_ids = set([sample["question_id"] for sample in val_questions])
revqa_questions_ids = set([sample["question_id"] for sample in revqa_questions])
overlap_question_ids = val_questions_ids.intersection(revqa_questions_ids)

questions = []
questions_ids = []
image_id_dict = defaultdict(list)
replace_words = ["?", "? ", " a ", " the ", " an "]
condensed_questions_counter = Counter()

for question in revqa_questions:
    if question["question_id"] in overlap_question_ids or question.get("rephrasing_of", -1) in overlap_question_ids:
        questions.append(question)
        questions_ids.append(question["question_id"])
        image_id_dict[question["image_id"]].append(question["question_id"])
        condensed_question = question["question"]
        # 139013 vs 139405
        # for word in replace_words:
        #     if replace_words in ["?", "? "]:
        #         condensed_question = condensed_question.replace(word, "")
        #     else:
        #         condensed_question = condensed_question.replace(word, " ")
        condensed_questions_counter[condensed_question] += 1

assert len(questions) == len(revqa_questions)

for question in questions:
    question["pos_ids"] = []
    if "rephrasing_of" in question:
        question["pos_ids"].append(question["rephrasing_of"])
        main_id = int(question["rephrasing_of"])
        for i in range(0, 3):
            qid = str(main_id*10 + i)
            question["pos_ids"].append(qid)
            assert qid in questions_ids
    else:
        question["pos_ids"].append(question["question_id"])
        main_id = int(question["question_id"])
        for i in range(0, 3):
            qid = str(main_id*10 + i)
            question["pos_ids"].append(qid)
            assert qid in questions_ids


questions = sorted(questions, key=lambda x: x["question_id"])

answers_val_dict = {}
for answer in answers_val:
    answers_val_dict[answer["question_id"]] = answer

answers = []
for question in questions:
    question_id = question["question_id"]

    if "rephrasing_of" in question:
        question_id = question["rephrasing_of"]

    answer = deepcopy(answers_val_dict[question_id])
    answer["question_id"] = question["question_id"]
    answers.append(answer)

for (q, a) in zip(questions, answers):
    assert q["question_id"] == a["question_id"]

# dump data
question_splits = [questions[:-20000], questions[-20000:]]
answer_splits = [answers[:-20000], answers[-20000:]]

for ans_split, que_split, ans_path, que_path, split in zip(answer_splits, question_splits, answer_paths, json_paths, ["train", "val"]):
    revqa_data = {
        "questions": que_split,
        "data_subtype": f"val2014_rephrasings_{split}",
    }
    json.dump(revqa_data, open(que_path, "w"))
    cPickle.dump(ans_split, open(ans_path, "wb"))

