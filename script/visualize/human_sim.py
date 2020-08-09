import json
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sacrebleu
import _pickle as cPickle
warnings.filterwarnings("ignore")

def cos_sim(A, B):
    # A and B -> [N x dim]
    Bt = np.transpose(B)
    AB_mul = np.matmul(A, Bt)
    A_mag = np.linalg.norm(A, axis=1, keepdims=True)
    B_mag = np.linalg.norm(B, axis=1, keepdims=True)
    AB_mag = np.matmul(A_mag, np.transpose(B_mag))
    assert AB_mag.shape == AB_mul.shape
    return AB_mul/AB_mag


def rephrasings_dict(questions):
    question_rephrase_dict = {}

    for question in questions:
        if "rephrasing_of" in question:
            question_rephrase_dict[question["question_id"]] = question["rephrasing_of"]
        elif "rephrasing_ids" in question:
            min_qid = min(question["rephrasing_ids"] + [question["question_id"]])
            question_rephrase_dict[question["question_id"]] = min_qid
        else:
            question_rephrase_dict[question["question_id"]] = question["question_id"]

    return question_rephrase_dict

paths_dict = {
    "re_train": ["../../data/re-vqa/data/revqa_train_proc.json", "../../datasets/VQA/cache/revqa_train_target.pkl", "train"],
    "re_val": ["../../data/re-vqa/data/revqa_val_proc.json", "../../datasets/VQA/cache/revqa_val_target.pkl", "val"],
}

questions, answers = [], []
for key, value in paths_dict.items():
    _questions, _answers = json.load(open(value[0]))["questions"], cPickle.load(open(value[1], "rb"))
    questions.extend(_questions)
    answers.extend(_answers)

questions = sorted(questions, key=lambda x: x["question_id"])
answers = sorted(answers, key=lambda x: x["question_id"])

question_rephrase_dict = rephrasings_dict(questions)

questions_dict = {}
for question, answer in zip(questions, answers):
    questions_dict[question["question_id"]] = question


df = pd.DataFrame(columns=['min-qid', 'reference', 'rephrasing-1', 'rephrasing-2', 'rephrasing-3'])

added = []
for idx, question in enumerate(questions):
    all_ids = question["rephrasing_ids"] + [question["question_id"]]
    all_ids = sorted(all_ids)
    min_qid = min(all_ids)
    if min_qid in added:
        continue
    else:
        added.append(min_qid)
    _questions = [questions_dict[qid]["question"] for qid in all_ids]
    try:
        df.loc[idx] = [min_qid] + _questions[:4]
    except:
        import pdb
        pdb.set_trace()

sim_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
data = []
col_names = list(df.keys())

for index, row in tqdm(df.iterrows(), total=len(df)):
    batch = list(row)
    batch = batch[1:]
    ref_question = batch[0]
    batch_embeddings = sim_model.encode(batch)
    sim_scores = cos_sim([batch_embeddings[0]], batch_embeddings).round(2)[0]
    # sort batch based on sim_scores
    question_sim_pairs = [(x, y) for x, y in zip(batch, sim_scores)]
    question_sim_pairs = [x[1] for x in question_sim_pairs]
    df.loc[index, "avg"] = sum(question_sim_pairs)/len(question_sim_pairs)
    data.append(question_sim_pairs)

import pdb
pdb.set_trace()

