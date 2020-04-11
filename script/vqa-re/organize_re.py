import json
import os
import _pickle as cPickle
from copy import deepcopy

# Todo:
#   - Build json and pkl files
from tqdm import tqdm
from collections import Counter

# Fixing collisions with question-ids

revqa_path = "../../data/re-vqa/data/v2_OpenEnded_mscoco_valrep2014_humans_og_questions.json"
org_revqa_path = "../../data/re-vqa/data/v2_OpenEnded_mscoco_valrep2014_humans_og_questions_organized.json"
revqa_data = json.load(open(revqa_path))

org_factor = int(1e10)

for question in tqdm(revqa_data["questions"]):
    if "rephrasing_of" in question:
        old_idx = question["question_id"]
        idx = int(question["question_id"] - question["rephrasing_of"] * 10)
        try:
            assert idx in [0, 1, 2, 3]
        except:
            import pdb
            pdb.set_trace()
        question["question_id"] = question["rephrasing_of"] * org_factor + idx
        # print(f"Replacing: {question['rephrasing_of']}, {question['question_id']}, {old_idx}")

json.dump(revqa_data, open(org_revqa_path, "w"))
